# SPDX-License-Identifier: Apache-2.0
"""Module accuracy test: Mistral MLP vs Hugging Face, prefill and decode.

Follows ``docs/design/accuracy/module_test_guidelines.md``: real checkpoint
weights for one layer, at least N=10 random inputs (more for small tensors,
see ``MIN_ERROR_VALUES``), a per-sample two-way check that only prints, and
the aggregated three-way check as the gate (FP32 HF baseline, BF16 HF
reference, BF16 vLLM Neuron target).

The MLP code is unchanged from Llama, so this mainly checks the Mistral
weight mapping and TP sharding (gate/up split on dim 1, down on dim 0) at
world sizes 1, 2 and 4. Expect the error to grow with TP (sigma-ratio about
1.0 / 1.1 / 1.2 at TP 1 / 2 / 4): each rank rounds its partial output to BF16
before the cross-rank sum, which a single-device BF16 reference does not.
A plain-PyTorch simulation of the TP-way BF16 split gives the same ratios.

``MISTRAL_MODEL`` must be a local checkpoint directory without
``consolidated.safetensors`` (the tiny checkpoint from
``test/unit/tiny_checkpoint.py`` or a filtered Mistral-7B-v0.3 download):

    pip install -e contrib/models/mistral
    # CPU, no Neuron device:
    VLLM_NEURON_CPU_MODE=1 MISTRAL_MODEL=/tmp/tiny-mistral \\
        pytest contrib/models/mistral/test/module/test_mlp.py -v -s --timeout=600
    # On a Neuron instance:
    MISTRAL_MODEL=/path/to/Mistral-7B-Instruct-v0.3 \\
        pytest contrib/models/mistral/test/module/test_mlp.py -v -s --timeout=3600
"""

import copy
import functools
import glob
import os

import pytest
import torch
from torch import nn

pytest.importorskip("vllm")
transformers = pytest.importorskip("transformers")

MODEL_PATH = os.environ.get("MISTRAL_MODEL")
CPU_MODE = os.environ.get("VLLM_NEURON_CPU_MODE") == "1"
DEVICE = "cpu" if CPU_MODE else "neuron:0"
LAYER = 0
# At least N_SAMPLES inputs per case, and more when the tensors are small, so
# every case pools at least MIN_ERROR_VALUES error values for the three-way
# statistics. The guidelines' N=10 assumes large tensors; a decode batch of 1
# on the tiny checkpoint (hidden 512) gives only 5,120 values at N=10, which
# makes BC noisy. The pass thresholds are unchanged.
N_SAMPLES = 10
MIN_ERROR_VALUES = 65_536
TWO_WAY_RTOL = 1e-2  # informational only, see module docstring

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.isdir(MODEL_PATH),
    reason="set MISTRAL_MODEL to a local Mistral checkpoint directory",
)


def _mlp_keys(layer: int) -> dict[str, str]:
    prefix = f"model.layers.{layer}.mlp"
    return {
        "gate_proj_weight": f"{prefix}.gate_proj.weight",
        "up_proj_weight": f"{prefix}.up_proj.weight",
        "down_proj_weight": f"{prefix}.down_proj.weight",
    }


def _load_hf_mlp(path: str, layer: int) -> nn.Module:
    """Load only layer ``layer``'s MLP from safetensors into the HF module."""
    from safetensors import safe_open
    from transformers.models.mistral.modeling_mistral import MistralMLP as HFMLP

    hf_config = transformers.AutoConfig.from_pretrained(path)
    module = HFMLP(hf_config)
    wanted = {v: k.replace("_weight", ".weight") for k, v in _mlp_keys(layer).items()}
    state = {}
    for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in wanted.keys() & set(f.keys()):
                state[wanted[key]] = f.get_tensor(key).to(torch.bfloat16)
    module.load_state_dict(state, strict=True)
    return module.to(torch.bfloat16).eval()


class _MLPRunner(nn.Module):
    """Fixes ``is_prefill`` so MPExecutor can call the MLP with one tensor."""

    def __init__(self, mlp: nn.Module, is_prefill: bool):
        super().__init__()
        self.mlp = mlp
        self.is_prefill = is_prefill

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states, is_prefill=self.is_prefill)


def _model_load(path: str, is_prefill: bool) -> nn.Module:
    """Runs in each MPExecutor worker after the parallel groups exist."""
    from vllm.distributed.parallel_state import get_tp_group

    from mistral_neuron.config import MistralConfig
    from mistral_neuron.model import MistralMLP
    from vllm_neuron import envs
    from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

    config = MistralConfig.from_configs(transformers.AutoConfig.from_pretrained(path))
    mlp = MistralMLP(config)

    tp = get_tp_group()
    checkpoint = SafetensorsCheckpoint(path)
    load = checkpoint.load_sharded if CPU_MODE else checkpoint.load_sharded_pipelined
    state = load(
        rank=tp.rank_in_group,
        world_size=tp.world_size,
        model=mlp,
        mappings=_mlp_keys(LAYER),
        device=torch.device(DEVICE),
    ).state_dict
    mlp.load_state_dict(state, strict=True, assign=True)

    runner = _MLPRunner(mlp, is_prefill).to(DEVICE).eval()
    if CPU_MODE:
        return runner
    return torch.compile(runner, backend=envs.get_compile_backend_name())


def _references(hf_mlp: nn.Module, x: torch.Tensor):
    with torch.no_grad():
        fp32 = copy.deepcopy(hf_mlp).float()(x.float())
        bf16 = hf_mlp(x)
    return fp32, bf16


def _two_way_note(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    rel = ((actual.float() - expected).abs().max() / expected.abs().max()).item()
    if rel > TWO_WAY_RTOL:
        print(f"NOTE: {name}: max relative error {rel:.4f} > {TWO_WAY_RTOL}")


def _run(world_size: int, is_prefill: bool, rows: int):
    from vllm_neuron.accuracy.testing import assert_close_three_way
    from vllm_neuron.utils.executor import MPExecutor

    hf_mlp = _load_hf_mlp(MODEL_PATH, LAYER)
    hidden = hf_mlp.gate_proj.in_features
    n_samples = max(N_SAMPLES, -(-MIN_ERROR_VALUES // (rows * hidden)))
    kind = "prefill" if is_prefill else "decode"
    name = f"mlp_{kind}/layer{LAYER}/rows{rows}/tp{world_size}"

    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(_model_load, MODEL_PATH, is_prefill),
        device=DEVICE,
    )
    all_fp32, all_bf16, all_target = [], [], []
    try:
        for sample in range(n_samples):
            torch.manual_seed(sample)
            x = torch.randn(rows, hidden, dtype=torch.bfloat16)
            fp32, bf16 = _references(hf_mlp, x)

            if is_prefill:
                # Sequence parallel: each rank gets its slice of the tokens and
                # returns the same slice of the output.
                chunks = x.chunk(world_size, dim=0)
                executor.dispatch_per_rank([(c,) for c in chunks])
                target = torch.cat(executor.collect(), dim=0)
            else:
                # Decode: every rank gets the full batch; the all-reduce
                # leaves the same output on every rank.
                executor.dispatch(x)
                outputs = executor.collect()
                for rank, out in enumerate(outputs[1:], start=1):
                    torch.testing.assert_close(out, outputs[0], rtol=0, atol=0,
                                               msg=f"rank {rank} differs from rank 0")
                target = outputs[0]

            assert target.shape == fp32.shape, (target.shape, fp32.shape)
            _two_way_note(f"{name}[{sample}]", target, fp32)
            all_fp32.append(fp32)
            all_bf16.append(bf16)
            all_target.append(target.float())
    finally:
        executor.shutdown()

    assert_close_three_way(all_fp32, all_bf16, all_target, name=name)


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_mlp_prefill(world_size, seq_len):
    _run(world_size, is_prefill=True, rows=seq_len)


# Largest intermediate size per rank that the plugin's fused decode MLP kernel
# (nkilib mlp_tkg) fits in SBUF on Trn2 at the default LNC=2. Mistral-7B-v0.3
# has 14336, so decode at world size 1 needs the kernel to hold twice this and
# does not compile; see the compatibility matrix in the README.
_MAX_DECODE_INTERMEDIATE_PER_RANK = 7168


def _decode_unsupported_on_device(world_size: int) -> str | None:
    if CPU_MODE:
        return None
    intermediate = transformers.AutoConfig.from_pretrained(MODEL_PATH).intermediate_size
    if intermediate // world_size <= _MAX_DECODE_INTERMEDIATE_PER_RANK:
        return None
    return (
        f"intermediate_size {intermediate} / TP {world_size} exceeds the "
        f"{_MAX_DECODE_INTERMEDIATE_PER_RANK} per rank the fused decode MLP kernel "
        "fits in SBUF on Trn2 (vllm-neuron 0.29.0.1.2.0); TP 1 decode did not "
        "work, see the compatibility matrix in the README"
    )


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_mlp_decode(world_size, batch_size):
    reason = _decode_unsupported_on_device(world_size)
    if reason:
        pytest.xfail(reason)
    _run(world_size, is_prefill=False, rows=batch_size)
