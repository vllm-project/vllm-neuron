# SPDX-License-Identifier: Apache-2.0
"""Full-model prefill regression test for BailingMoeV2 (inclusionAI/Ling-mini-2.0).

An earlier port revision gave NaN or inf logits at step 0 (the prefill) for both
prompts of the integration test. The component tests passed. Thus the defect
needs the full model with real activations: 20 layers, the real token IDs of
the logit prompts, and pad token 0 in the rest of the 256-token bucket.

This test loads the full model through ``load_weights`` on 8 ranks (TP 8, EP
8, 2 trn2.48xlarge nodes) and runs one prefill for each case:

  - The two logit prompts (``PROMPTS_2``), with the tokenization of the goldens.
  - Each pad position mode of the runner: the pads repeat the last position,
    or the pads are at position 0.

The probe records each stage of the SP shard of each rank: the embedding,
each attention residual, each MLP residual and the final norm. The test
asserts that each row of each stage is finite, and that the logits of the
last real token are finite. A failure gives the first stage with a NaN or inf
value, for the real rows and for the pad rows, and the peak values before
that stage.

The test also asserts that the prefill graph of each rank contains the NKI
kernels of the attention, the MoE and the dense MLP.

``test_full_model_server_forward_finite`` runs the same cases through the
model forward of the logit server: the server configs, the on-device sampler
and the logit all-gather. An earlier port revision gave finite logits in the probe
and NaN or inf logits in the server. This test finds if the server graph
causes the difference.

``test_full_model_server_forward_meta_init_finite`` also builds the model in a
``torch.device("meta")`` context, as the runner does. An earlier port revision
passed the server-forward test and gave NaN for all logits in the server. In
the meta context the RoPE head permutation of the loader was a meta tensor.

This test does not replace the integration test. That test compares the logits
with the goldens.

Run on each node through the launcher, which sets NODE_RANK and MASTER_ADDR:
    python3 test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py \
        -m pytest \
        contrib/models/bailing_moe/test/integration/test_full_model_prefill.py -v
"""

import importlib.util
from functools import partial
from pathlib import Path

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.utils.executor import MPExecutor

# This folder's own code loads by file location, so the folder keeps working
# when it is copied out of the repository. See ``test/_local_src.py``.
_LOADER = importlib.util.spec_from_file_location(
    "contrib_bailing_moe_local_src",
    Path(__file__).resolve().parents[1] / "_local_src.py",
)
_local_src = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(_local_src)

utils = _local_src.load_test_module("utils")
PROMPTS_2 = utils.PROMPTS_2

pytestmark = [pytest.mark.full_model_test]

RANDOM_SEED = 42
PREFILL_TOKENS = 256
PAD_TOKEN_ID = 0
# The KV cache of the sequence starts at block 1, because block 0 is the vLLM
# null block. Pad tokens write to slot 0.
FIRST_BLOCK = 1
NULL_SLOT = 0
NULL_BLOCK_SENTINEL = -1
BLOCKS_PER_SEQ = PREFILL_TOKENS // utils.BLOCK_SIZE
PAD_POSITION_MODES = ("repeat_last", "zero")
PREFILL_NKI_SITES = (
    utils.REQUIRED_NKI_SITES["attention prefill"]
    | utils.REQUIRED_NKI_SITES["moe prefill"]
    | utils.REQUIRED_NKI_SITES["dense mlp prefill"]
)


@pytest.fixture(autouse=True)
def require_device_mode():
    if envs.VLLM_NEURON_CPU_MODE:
        pytest.fail(
            "test_full_model_prefill.py is a device test. Unset "
            "VLLM_NEURON_CPU_MODE, and run it on 2 Neuron hosts."
        )
    torch.manual_seed(RANDOM_SEED)


def _prompt_input_ids(checkpoint: str) -> list[list[int]]:
    """Tokenize ``PROMPTS_2`` with the call of the golden generation."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return [
        tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)[
            "input_ids"
        ][0].tolist()
        for prompt in PROMPTS_2
    ]


def _positions(num_real: int, mode: str) -> torch.Tensor:
    fill = num_real - 1 if mode == "repeat_last" else 0
    positions = torch.full((PREFILL_TOKENS,), fill, dtype=torch.int32)
    positions[:num_real] = torch.arange(num_real, dtype=torch.int32)
    return positions


def _metadata(num_layers: int, num_real: int) -> dict:
    """Return the prefill metadata of the runner for one short sequence."""
    slots = torch.full((PREFILL_TOKENS,), NULL_SLOT, dtype=torch.int64)
    slots[:num_real] = FIRST_BLOCK * utils.BLOCK_SIZE + torch.arange(num_real)
    table = torch.full((1, BLOCKS_PER_SEQ), NULL_BLOCK_SENTINEL, dtype=torch.int32)
    table[0, 0] = FIRST_BLOCK
    layer = {
        "max_query_len": PREFILL_TOKENS,
        "decode_token_threshold": 1,
        "slot_mapping": slots,
        "block_size": utils.BLOCK_SIZE,
        "block_table_tensor": table,
        "cached_seq_len": None,
        "kv_segment_size": None,
    }
    return {f"layers.{i}.self_attn": dict(layer) for i in range(num_layers)}


def _first_bad_stage(flags: torch.Tensor, rows: torch.Tensor) -> int | None:
    """Return the first stage with a non-finite value in ``rows``, or None."""
    if rows.numel() == 0:
        return None
    bad = flags[:, rows].any(dim=1).nonzero()
    return None if bad.numel() == 0 else int(bad[0, 0])


def _describe(
    case: str,
    stages: list[str],
    flags: torch.Tensor,
    peaks: torch.Tensor,
    real_rows: torch.Tensor,
    pad_rows: torch.Tensor,
) -> str | None:
    """Return the failure text of one case, or None if each stage is finite."""
    first_real = _first_bad_stage(flags, real_rows)
    first_pad = _first_bad_stage(flags, pad_rows)
    if first_real is None and first_pad is None:
        return None

    def name(stage):
        return "none" if stage is None else stages[stage]

    first = min(s for s in (first_real, first_pad) if s is not None)
    lines = [
        f"{case}: first non-finite stage: real rows {name(first_real)}, "
        f"pad rows {name(first_pad)}."
    ]
    for stage in range(max(0, first - 3), first + 1):
        real_peak = peaks[stage, real_rows].max().item() if real_rows.numel() else 0.0
        pad_peak = peaks[stage, pad_rows].max().item() if pad_rows.numel() else 0.0
        lines.append(
            f"  {stages[stage]}: peak |x| real {real_peak:.4g}, pad {pad_peak:.4g}, "
            f"non-finite rows {int(flags[stage].sum())}"
        )
    return "\n".join(lines)


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_full_model_prefill_finite(world_size, tmp_path):
    """Run the full-model prefill of the logit prompts, and assert finite stages."""
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    checkpoint = utils.resolve_checkpoint()
    num_layers = utils.read_hf_config(checkpoint)["num_hidden_layers"]
    stages = utils.prefill_stage_names(num_layers)
    shard = PREFILL_TOKENS // world_size
    # The rows of this node, in the order of the local ranks.
    node_rows = torch.cat(
        [torch.arange(r * shard, (r + 1) * shard) for r in global_ranks]
    )
    prompts_input_ids = _prompt_input_ids(checkpoint)

    route_dir = tmp_path / "nki_routes"
    route_dir.mkdir()
    route_dir = str(route_dir)
    device = "neuron:0"
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            utils.full_model_prefill_load,
            checkpoint=checkpoint,
            cpu_mode=False,
            device=device,
            route_dir=route_dir,
        ),
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        ep_degree=utils.EP_DEGREE,
        load_timeout=7200,
    )

    failures = []
    routes_checked = False
    for prompt_idx, ids in enumerate(prompts_input_ids):
        num_real = len(ids)
        assert num_real < PREFILL_TOKENS, (
            f"Prompt {prompt_idx} has {num_real} tokens. The bucket has "
            f"{PREFILL_TOKENS} tokens, so the case has no pad row."
        )
        input_ids = torch.full((PREFILL_TOKENS,), PAD_TOKEN_ID, dtype=torch.int64)
        input_ids[:num_real] = torch.tensor(ids, dtype=torch.int64)
        for mode in PAD_POSITION_MODES:
            case = f"prompt {prompt_idx} ({num_real} tokens, pads {mode})"
            executor.dispatch(
                input_ids,
                _positions(num_real, mode),
                _metadata(num_layers, num_real),
                torch.tensor([num_real - 1], dtype=torch.int64),
            )
            outputs = executor.collect(timeout=7200)
            assert len(outputs) == len(global_ranks)

            if not routes_checked:
                records = utils.read_route_records(route_dir, global_ranks)
                utils.assert_nki_sites(
                    records, {}, "full-model prefill", required=PREFILL_NKI_SITES
                )
                routes_checked = True

            flags = torch.cat([o[0].cpu() for o in outputs], dim=1)
            peaks = torch.cat([o[1].cpu() for o in outputs], dim=1)
            is_real = node_rows < num_real
            real_rows = is_real.nonzero().flatten()
            pad_rows = (~is_real).nonzero().flatten()
            text = _describe(case, stages, flags, peaks, real_rows, pad_rows)

            logits = outputs[0][2].cpu()
            bad_logits = int((~torch.isfinite(logits)).sum())
            for rank_out in outputs[1:]:
                torch.testing.assert_close(
                    rank_out[2].cpu(),
                    logits,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                    msg=f"{case}: the ranks give different logits.",
                )
            print(
                f"[bailing prefill] node {node_rank} {case}: "
                f"{'finite' if text is None else 'NON-FINITE'}, "
                f"non-finite logits {bad_logits}, "
                f"final norm peak {peaks[-1].max().item():.4g}"
            )
            if text is not None:
                failures.append(text)
            if bad_logits:
                failures.append(
                    f"{case}: {bad_logits} of {logits.numel()} logits of the last "
                    "real token are NaN or inf."
                )

    assert not failures, (
        f"The full-model prefill on node {node_rank} gives NaN or inf values.\n"
        + "\n".join(failures)
    )


# Runner sampling parameters of one greedy request: [top_k, top_p, temperature].
GREEDY_SAMPLING_PARAMS = [[0, 1.0, 1.0]]


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_full_model_server_forward_finite(world_size, tmp_path):
    """Run the model forward of the logit server, and assert finite logits.

    An earlier port revision: ``test_full_model_prefill_finite`` gave finite logits
    for each case, but the logit server gave NaN or inf logits at step 0 for
    both prompts. That probe runs the modules one by one, without the sampler,
    with the ``config.json`` dict. This test removes these differences. It
    builds the model with the configs of the server, and it calls
    ``BailingMoeV2ForCausalLM.forward`` with the inputs of the runner. The
    graph thus contains the on-device sampler and the logit all-gather.

    The cases are the same as in ``test_full_model_prefill_finite``. For each
    case the test asserts these conditions:

      - The gathered logits are finite.
      - All ranks give the same logits and the same sampled token.
      - The sampled token has the largest logit.

    Before the dispatch, the test also asserts that the server config path
    and the ``config.json`` path give the same model config.
    """
    _run_server_forward(world_size, tmp_path, meta_init=False)


@pytest.mark.parametrize("world_size", [utils.WORLD_SIZE])
@pytest.mark.usefixtures("dist_cleanup")
def test_full_model_server_forward_meta_init_finite(world_size, tmp_path):
    """Run the server forward with the meta construction of the runner.

    An earlier port revision: ``test_full_model_server_forward_finite`` passed, but
    the logit server gave NaN for all 157184 logits of each step. The runner
    builds the model in a ``torch.device("meta")`` context, and that test
    builds it on CPU. In the meta context the RoPE head permutation of the
    loader was a meta tensor, so the Q and K weights had no values.

    This test builds the model with the steps of the runner (see
    ``utils.load_like_runner``). The cases and the assertions are the same as
    in ``test_full_model_server_forward_finite``.
    """
    _run_server_forward(world_size, tmp_path, meta_init=True)


def _run_server_forward(world_size: int, tmp_path, meta_init: bool) -> None:
    nnodes, node_rank = utils.device_topology()
    global_ranks = utils.local_global_ranks(nnodes, node_rank)
    checkpoint = utils.resolve_checkpoint()
    differences = utils.config_path_differences(checkpoint)
    assert not differences, (
        "The server config path gives a different model config from config.json:\n"
        + "\n".join(differences)
    )
    num_layers = utils.read_hf_config(checkpoint)["num_hidden_layers"]
    prompts_input_ids = _prompt_input_ids(checkpoint)

    route_dir = tmp_path / "nki_routes"
    route_dir.mkdir()
    route_dir = str(route_dir)
    device = "neuron:0"
    executor = MPExecutor(
        world_size=world_size,
        model_load=partial(
            utils.full_model_server_forward_load,
            checkpoint=checkpoint,
            cpu_mode=False,
            device=device,
            route_dir=route_dir,
            meta_init=meta_init,
        ),
        device=device,
        nnodes=nnodes,
        node_rank=node_rank,
        ep_degree=utils.EP_DEGREE,
        load_timeout=7200,
    )

    failures = []
    routes_checked = False
    for prompt_idx, ids in enumerate(prompts_input_ids):
        num_real = len(ids)
        assert num_real < PREFILL_TOKENS, (
            f"Prompt {prompt_idx} has {num_real} tokens. The bucket has "
            f"{PREFILL_TOKENS} tokens, so the case has no pad row."
        )
        input_ids = torch.full((PREFILL_TOKENS,), PAD_TOKEN_ID, dtype=torch.int64)
        input_ids[:num_real] = torch.tensor(ids, dtype=torch.int64)
        for mode in PAD_POSITION_MODES:
            case = (
                f"server forward (meta_init={meta_init}) prompt {prompt_idx} "
                f"({num_real} tokens, pads {mode})"
            )
            executor.dispatch(
                input_ids,
                _positions(num_real, mode),
                _metadata(num_layers, num_real),
                torch.tensor([num_real - 1], dtype=torch.int64),
                torch.tensor(GREEDY_SAMPLING_PARAMS, dtype=torch.float32),
            )
            outputs = executor.collect(timeout=7200)
            assert len(outputs) == len(global_ranks)

            if not routes_checked:
                records = utils.read_route_records(route_dir, global_ranks)
                utils.assert_nki_sites(
                    records, {}, "server forward", required=PREFILL_NKI_SITES
                )
                routes_checked = True

            tokens = outputs[0][0].cpu().flatten()
            logits = outputs[0][1].cpu()
            assert logits.shape == (1, utils.VOCAB_SIZE), (
                f"{case}: the gathered logits have shape {tuple(logits.shape)}. "
                f"The server gives (1, {utils.VOCAB_SIZE})."
            )
            for rank_out in outputs[1:]:
                torch.testing.assert_close(
                    rank_out[1].cpu(),
                    logits,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                    msg=f"{case}: the ranks give different logits.",
                )
                assert torch.equal(rank_out[0].cpu().flatten(), tokens), (
                    f"{case}: the ranks give different sampled tokens."
                )

            is_nan = torch.isnan(logits)
            is_posinf = logits == float("inf")
            is_neginf = logits == float("-inf")
            bad = is_nan | is_posinf | is_neginf
            shard = utils.VOCAB_SIZE // world_size
            bad_shards = sorted(
                {int(i) // shard for i in bad.flatten().nonzero().flatten()}
            )
            print(
                f"[bailing server forward] node {node_rank} {case}: "
                f"sampled token {tokens.tolist()}, NaN {int(is_nan.sum())}, "
                f"+inf {int(is_posinf.sum())}, -inf {int(is_neginf.sum())}, "
                f"vocab shards with a non-finite logit {bad_shards}"
            )
            if bad.any():
                failures.append(
                    f"{case}: {int(bad.sum())} of {logits.numel()} gathered logits "
                    f"are NaN or inf (NaN {int(is_nan.sum())}, +inf "
                    f"{int(is_posinf.sum())}, -inf {int(is_neginf.sum())}). The "
                    f"vocab shards with a non-finite logit are {bad_shards}."
                )
                continue
            token = int(tokens[0])
            if logits[0, token] != logits.max():
                failures.append(
                    f"{case}: the sampled token {token} has logit "
                    f"{logits[0, token].item():.6g}. The largest logit is "
                    f"{logits.max().item():.6g} (token {int(logits.argmax())})."
                )

    assert not failures, (
        f"The server forward on node {node_rank} gives incorrect results.\n"
        + "\n".join(failures)
    )
