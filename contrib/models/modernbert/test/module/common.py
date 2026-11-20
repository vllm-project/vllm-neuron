# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the ModernBERT (granite-embedding-small-english-r2) tests.

The port side runs in MPExecutor workers. Each ``model_load_*`` function is a
top-level function, so ``functools.partial`` of it is picklable and the worker
pool can reuse its processes.

The reference side is the HF transformers ``ModernBertModel`` in eager mode,
in FP32 (baseline) and BF16 (expected). The component references replay the
HF submodule calls that a forward pre-hook records. This keeps the reference
independent of the HF version-specific call signatures.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from vllm_neuron import envs
from vllm_neuron.model.neuron_config import NeuronConfig

# Load this folder's own ``src/`` by file location, not through a package path
# rooted at the repository: a contrib model has to keep working when the folder
# is copied into an application. ``test/_local_src.py`` is the loader, and it is
# loaded the same way, under a name no other contrib model can collide with.
_LOADER_NAME = "contrib_modernbert_local_src"
if _LOADER_NAME in sys.modules:
    _local_src = sys.modules[_LOADER_NAME]
else:
    _loader_spec = importlib.util.spec_from_file_location(
        _LOADER_NAME, Path(__file__).resolve().parents[1] / "_local_src.py"
    )
    _local_src = importlib.util.module_from_spec(_loader_spec)
    sys.modules[_LOADER_NAME] = _local_src
    _loader_spec.loader.exec_module(_local_src)

_src_config = _local_src.load("config")
_src_model = _local_src.load("model")

ModernBertConfig = _src_config.ModernBertConfig
ModernBertEncoderModel = _src_model.ModernBertEncoderModel
modernbert_attention_bounds = _src_model.modernbert_attention_bounds

logger = logging.getLogger(__name__)

MODEL_ID = "ibm-granite/granite-embedding-small-english-r2"
MODEL_REVISION = "2ab6fa8ea2d674564defd37171ae19079b864b33"
CHECKPOINT_ENV = "MODERNBERT_CHECKPOINT"

# Tensor parallel degree of the port. The model has 12 heads and an
# intermediate size of 1536, so TP 2 gives 6 heads and 768 channels per rank.
TP_SIZE = 2

SEED = 42

# Text for the 1024-token inputs. The text repeats until the tokenizer
# truncates it, so the input has real token statistics.
_TEXT_A = (
    "Granite embedding models encode a query and a passage into one vector "
    "each. A retrieval system ranks the passages by the cosine similarity of "
    "the vectors. The encoder reads the full input in both directions, and a "
    "sliding window limits the attention span of most layers. "
)
_TEXT_B = (
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose "
    "and oxygen. The chloroplast holds the pigments that absorb the light. "
    "def add(a, b):\n    return a + b  # simple addition\n"
    "Order #4521 shipped on 2024-03-12 via carrier ID 7F-A9; ETA 3-5 days. "
)


# =============================================================================
# Checkpoint
# =============================================================================


def resolve_checkpoint() -> str:
    """Return the local directory of the original checkpoint.

    ``MODERNBERT_CHECKPOINT`` selects a local snapshot of the original HF
    checkpoint. If it is not set, this downloads the pinned revision of the
    public model ID from the Hugging Face Hub.
    """
    path = os.environ.get(CHECKPOINT_ENV)
    if path:
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise FileNotFoundError(
                f"{CHECKPOINT_ENV}={path!r} has no config.json. "
                "Set it to the directory of the original HF checkpoint."
            )
        source = f"{CHECKPOINT_ENV} (revision {MODEL_REVISION})"
    else:
        # The plugin's shared checkpoint resolver lives outside this folder, so
        # the contrib tests download the pinned revision from the Hub instead.
        from huggingface_hub import snapshot_download

        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        source = f"snapshot_download({MODEL_ID!r}, revision={MODEL_REVISION!r})"
    logger.info("ModernBERT checkpoint: %s, from %s", path, source)
    return path


def load_config(checkpoint_path: str) -> ModernBertConfig:
    """Return the port config of the checkpoint."""
    return ModernBertConfig.from_configs(
        os.path.join(checkpoint_path, "config.json"), NeuronConfig()
    )


# =============================================================================
# Inputs
# =============================================================================


def make_token_ids(checkpoint_path: str, lengths: list[int]) -> list[torch.Tensor]:
    """Return one ``[length]`` int64 token-id tensor for each length.

    Each sequence starts with the CLS token and ends with the SEP token, as
    the tokenizer gives them. The sequences use different texts.
    """
    from transformers import AutoTokenizer

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    texts = [_TEXT_A, _TEXT_B]
    out = []
    for i, length in enumerate(lengths):
        text = texts[i % len(texts)] * (length // 8 + 1)
        ids = tokenizer(
            text, truncation=True, max_length=length, add_special_tokens=True
        )["input_ids"]
        if len(ids) != length:
            raise ValueError(
                f"The tokenizer gave {len(ids)} tokens, but the test needs "
                f"{length}. Make the input text longer."
            )
        out.append(torch.tensor(ids, dtype=torch.int64))
    return out


def pack(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the sequences into flat ``input_ids`` and ``positions``, as the runner does."""
    input_ids = torch.cat(sequences)
    positions = torch.cat(
        [torch.arange(s.shape[0], dtype=torch.int64) for s in sequences]
    )
    return input_ids, positions


# =============================================================================
# HF reference
# =============================================================================


def load_hf_models(checkpoint_path: str) -> tuple[nn.Module, nn.Module]:
    """Return the HF ``ModernBertModel`` in FP32 and in BF16 (eager attention)."""
    from transformers import AutoModel

    models = []
    for dtype in (torch.float32, torch.bfloat16):
        model = AutoModel.from_pretrained(
            checkpoint_path, torch_dtype=dtype, attn_implementation="eager"
        )
        models.append(model.eval())
    return models[0], models[1]


@torch.no_grad()
def hf_hidden_states(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the HF last hidden state ``[S, H]`` of one sequence."""
    ids = input_ids.unsqueeze(0)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    return out.last_hidden_state[0]


def cls_embedding(hidden_states: torch.Tensor) -> torch.Tensor:
    """Return the CLS-pooled and L2-normalized embedding ``[H]`` in FP32."""
    return F.normalize(hidden_states[0].float(), p=2, dim=-1)


def _cast(value, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, (tuple, list)):
        return type(value)(_cast(v, dtype) for v in value)
    if isinstance(value, dict):
        return {k: _cast(v, dtype) for k, v in value.items()}
    return value


@torch.no_grad()
def record_hf_calls(
    model: nn.Module, module_names: list[str], input_ids: torch.Tensor
) -> dict[str, tuple[tuple, dict]]:
    """Run the HF model and record the ``(args, kwargs)`` of each named submodule."""
    records = {}
    handles = []
    for name in module_names:

        def pre_hook(module, args, kwargs, _name=name):
            records[_name] = (tuple(args), dict(kwargs))

        handles.append(
            model.get_submodule(name).register_forward_pre_hook(
                pre_hook, with_kwargs=True
            )
        )
    try:
        hf_hidden_states(model, input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return records


@torch.no_grad()
def replay_hf_call(
    module: nn.Module,
    record: tuple[tuple, dict],
    hidden_states: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Call ``module`` with the recorded arguments and a new first input.

    Returns the first output ``[S, H]`` of batch item 0.
    """
    args, kwargs = record
    args = (hidden_states.to(dtype),) + tuple(_cast(a, dtype) for a in args[1:])
    out = module(*args, **_cast(kwargs, dtype))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out[0]


def hf_component_references(
    checkpoint_path: str, module_name: str, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input, fp32_out, bf16_out)`` for one HF submodule.

    ``input`` ``[S, H]`` BF16 is the input that the BF16 HF model gives to the
    submodule. The FP32 and BF16 outputs come from the same input.
    """
    hf_fp32, hf_bf16 = load_hf_models(checkpoint_path)
    rec_fp32 = record_hf_calls(hf_fp32, [module_name], input_ids)[module_name]
    rec_bf16 = record_hf_calls(hf_bf16, [module_name], input_ids)[module_name]
    hidden = rec_bf16[0][0].to(torch.bfloat16)
    out_fp32 = replay_hf_call(
        hf_fp32.get_submodule(module_name), rec_fp32, hidden, torch.float32
    )
    out_bf16 = replay_hf_call(
        hf_bf16.get_submodule(module_name), rec_bf16, hidden, torch.bfloat16
    )
    return hidden[0], out_fp32.float(), out_bf16.float()


# =============================================================================
# Port side (runs in the MPExecutor workers)
# =============================================================================


def _load_backbone(checkpoint_path: str) -> ModernBertEncoderModel:
    """Build the backbone for this rank and load the TP shard of the checkpoint."""
    model = ModernBertEncoderModel(load_config(checkpoint_path))
    model.load_weights(
        checkpoint_path,
        torch.device("cpu"),
        None,
        dist.get_rank(),
        dist.get_world_size(),
        pipelined=False,
    )
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_meta:
            raise RuntimeError(f"Parameter {name} is on the meta device after loading.")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(f"Parameter {name} has a value that is not finite.")
    return model.eval()


class FullModelRunner(nn.Module):
    """Run the backbone. Returns the post-norm hidden states ``[T, H]``."""

    def __init__(self, backbone: ModernBertEncoderModel):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.backbone(input_ids, positions)


class AttentionRunner(nn.Module):
    """Run the attention of one layer on a normalized input ``[T, H]``."""

    def __init__(self, backbone: ModernBertEncoderModel, layer_idx: int):
        super().__init__()
        self.attn = backbone.layers[layer_idx].attn
        self.window = backbone.config.sliding_window

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        full_min, full_max, sliding_min, sliding_max = modernbert_attention_bounds(
            positions, self.window
        )
        if self.attn.is_sliding:
            return self.attn(hidden_states, positions, sliding_min, sliding_max)
        return self.attn(hidden_states, positions, full_min, full_max)


class LayerRunner(nn.Module):
    """Run one encoder layer (attention and MLP with the residuals)."""

    def __init__(self, backbone: ModernBertEncoderModel, layer_idx: int):
        super().__init__()
        self.layer = backbone.layers[layer_idx]
        self.window = backbone.config.sliding_window

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        full_min, full_max, sliding_min, sliding_max = modernbert_attention_bounds(
            positions, self.window
        )
        if self.layer.attn.is_sliding:
            return self.layer(hidden_states, positions, sliding_min, sliding_max)
        return self.layer(hidden_states, positions, full_min, full_max)


class MLPRunner(nn.Module):
    """Run the GeGLU MLP of one layer on a normalized input ``[T, H]``."""

    def __init__(self, backbone: ModernBertEncoderModel, layer_idx: int):
        super().__init__()
        self.mlp = backbone.layers[layer_idx].mlp

    def forward(self, hidden_states: torch.Tensor):
        return self.mlp(hidden_states)


class ShardDumpRunner(nn.Module):
    """Return the loaded shard of each parameter, in sorted-name order."""

    def __init__(self, backbone: ModernBertEncoderModel):
        super().__init__()
        self.backbone = backbone

    def forward(self, _unused: torch.Tensor):
        params = dict(self.backbone.named_parameters())
        return tuple(params[name].detach().clone() for name in sorted(params))


_RUNNERS = {
    "full": lambda backbone, layer_idx: FullModelRunner(backbone),
    "attention": AttentionRunner,
    "layer": LayerRunner,
    "mlp": MLPRunner,
    "shards": lambda backbone, layer_idx: ShardDumpRunner(backbone),
}


def model_load(
    checkpoint_path: str, component: str, layer_idx: int, device: str
) -> nn.Module:
    """MPExecutor ``model_load``: build one runner on this rank.

    On CPU, return the eager module. On Neuron, compile the module with the
    native backend.
    """
    backbone = _load_backbone(checkpoint_path)
    runner = _RUNNERS[component](backbone, layer_idx).eval()
    if device == "cpu":
        return runner
    runner = runner.to(device)
    return torch.compile(
        runner, fullgraph=True, backend=envs.get_compile_backend_name()
    )


def expected_param_names(config: ModernBertConfig) -> list[str]:
    """Return the sorted parameter names of ``ModernBertEncoderModel``."""
    names = [
        "embeddings.tok_embeddings.weight",
        "embeddings.norm.weight",
        "final_norm.weight",
    ]
    for i in range(config.num_hidden_layers):
        if i != 0:
            names.append(f"layers.{i}.attn_norm.weight")
        names += [
            f"layers.{i}.attn.qkv_proj_weight",
            f"layers.{i}.attn.o_proj_weight",
            f"layers.{i}.mlp_norm.weight",
            f"layers.{i}.mlp.gate_proj_weight",
            f"layers.{i}.mlp.up_proj_weight",
            f"layers.{i}.mlp.down_proj_weight",
        ]
    return sorted(names)
