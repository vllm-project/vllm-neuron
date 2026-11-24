# SPDX-License-Identifier: Apache-2.0
"""Component checks of the Ouro modules against the independent reference.

Each ``check_*`` function builds one real Ouro module in ``MPExecutor``
workers, runs it, and compares the result with ``ouro_reference.py``. A
three-way check (fp32 reference, bf16 reference, Neuron) applies with the
default thresholds of ``assert_close_three_way``.

The same checks run with two checkpoints:

* ``test/unit/test_components_cpu.py``: the tiny random-weight checkpoint,
  TP 2, CPU mode.
* ``test/unit/test_checkpoint_cpu.py`` and
  ``test/module/test_components_device.py``: the real Ouro-1.4B weights.

The attention checks run on one UT step, and check that the KV caches of the
other UT steps stay zero.
"""

import functools

import torch

from vllm_neuron.accuracy.testing import assert_close_three_way
from vllm_neuron.utils.executor import MPExecutor

from _local_src import src_module
from ouro_reference import (
    layer_keys,
    load_tensors,
    read_config,
    ref_attention,
    ref_decoder_layer,
    ref_exit_select,
    ref_mlp,
)
from ouro_test_utils import (
    GATE_PREFIX,
    gather_cache_tokens,
    load_attention_harness,
    load_decoder_harness,
    load_kv_spec_harness,
    load_mlp_harness,
    load_weight_check_harness,
    ouro_config,
    prefill_metadata,
)

COLLECT_TIMEOUT = 3600


def _weights(model_checkpoint: str, layer_idx: int):
    keys = layer_keys(layer_idx)
    return (
        load_tensors(model_checkpoint, keys, torch.float32),
        load_tensors(model_checkpoint, keys, torch.bfloat16),
    )


def _rope(config, positions: torch.Tensor):
    OuroRotaryEmbedding = src_module("model").OuroRotaryEmbedding

    return OuroRotaryEmbedding(config)(
        positions.to(torch.int32), device=torch.device("cpu"), dtype=torch.bfloat16
    )


def _layer_name(config, ut_step: int, layer_idx: int) -> str:
    kv_layer_name = src_module("model").kv_layer_name

    return kv_layer_name(ut_step, layer_idx, config.num_hidden_layers)


def _assert_other_slots_zero(outputs: list, index: int = 3) -> None:
    for rank, o in enumerate(outputs):
        assert torch.all(o[index] == 0), (
            f"Rank {rank} wrote into the KV cache of another UT step: "
            f"{o[index].tolist()}."
        )


def check_decoder_layer_prefill(
    model_checkpoint: str,
    world_size: int,
    device: str,
    layer_idx: int,
    ut_step: int,
    seq_len: int,
    block_size: int,
) -> None:
    """One decoder layer with sandwich norms, on one UT step, with SP."""
    config = ouro_config(model_checkpoint)
    raw = read_config(model_checkpoint)
    w32, w16 = _weights(model_checkpoint, layer_idx)

    torch.manual_seed(42)
    x = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(seq_len, dtype=torch.int64)
    ref32, (k32, v32) = ref_decoder_layer(
        x.float().unsqueeze(0), positions, w32, layer_idx, raw
    )
    ref16, (k16, v16) = ref_decoder_layer(
        x.unsqueeze(0), positions, w16, layer_idx, raw
    )

    num_blocks = 1 + seq_len // block_size
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_decoder_harness,
            model_checkpoint,
            layer_idx,
            ut_step,
            num_blocks,
            block_size,
            device=device,
        ),
        device=device,
    )
    cos, sin = _rope(config, positions)
    layer_name = _layer_name(config, ut_step, layer_idx)
    metadata = prefill_metadata(layer_name, seq_len, block_size)
    chunks = x.chunk(world_size, dim=0)
    executor.dispatch_per_rank(
        [(chunk, positions.to(torch.int32), cos, sin, metadata) for chunk in chunks]
    )
    outputs = executor.collect(timeout=COLLECT_TIMEOUT)

    out = torch.cat([o[0].cpu() for o in outputs], dim=0)
    assert_close_three_way(
        baseline=ref32[0],
        expected=ref16[0],
        actual=out,
        name=f"decoder_prefill/layer{layer_idx}/ut{ut_step}/tp{world_size}",
    )

    slots = metadata[layer_name]["slot_mapping"]
    k_cache = gather_cache_tokens([o[1].cpu() for o in outputs], slots, block_size)
    v_cache = gather_cache_tokens([o[2].cpu() for o in outputs], slots, block_size)
    assert_close_three_way(
        baseline=k32[0], expected=k16[0], actual=k_cache, name="decoder_prefill/k"
    )
    assert_close_three_way(
        baseline=v32[0], expected=v16[0], actual=v_cache, name="decoder_prefill/v"
    )

    _assert_other_slots_zero(outputs)
    for rank, o in enumerate(outputs):
        assert torch.all(o[1][0] == 0), f"Rank {rank} wrote into the null block."


def check_attention_flash_prefill(
    model_checkpoint: str,
    world_size: int,
    device: str,
    layer_idx: int,
    ut_step: int,
    seq_len: int,
    block_size: int,
) -> None:
    """Flash-attention prefill of one layer on one UT step."""
    config = ouro_config(model_checkpoint)
    raw = read_config(model_checkpoint)
    w32, w16 = _weights(model_checkpoint, layer_idx)
    prefix = f"model.layers.{layer_idx}.self_attn"

    torch.manual_seed(42)
    h = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(seq_len, dtype=torch.int64)
    ref32, (k32, v32) = ref_attention(
        h.float().unsqueeze(0), positions, w32, prefix, raw
    )
    ref16, (k16, v16) = ref_attention(h.unsqueeze(0), positions, w16, prefix, raw)

    num_blocks = 1 + seq_len // block_size
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_attention_harness,
            model_checkpoint,
            layer_idx,
            ut_step,
            num_blocks,
            block_size,
            device,
        ),
        device=device,
    )
    cos, sin = _rope(config, positions)
    layer_name = _layer_name(config, ut_step, layer_idx)
    metadata = prefill_metadata(layer_name, seq_len, block_size)
    executor.dispatch_per_rank(
        [
            (chunk, positions.to(torch.int32), cos, sin, metadata)
            for chunk in h.chunk(world_size, dim=0)
        ]
    )
    outputs = executor.collect(timeout=COLLECT_TIMEOUT)

    assert_close_three_way(
        baseline=ref32[0],
        expected=ref16[0],
        actual=torch.cat([o[0].cpu() for o in outputs], dim=0),
        name=f"attn_flash_prefill/seq{seq_len}/tp{world_size}",
    )
    slots = metadata[layer_name]["slot_mapping"]
    assert_close_three_way(
        baseline=k32[0],
        expected=k16[0],
        actual=gather_cache_tokens([o[1].cpu() for o in outputs], slots, block_size),
        name="attn_flash_prefill/k_cache",
    )
    assert_close_three_way(
        baseline=v32[0],
        expected=v16[0],
        actual=gather_cache_tokens([o[2].cpu() for o in outputs], slots, block_size),
        name="attn_flash_prefill/v_cache",
    )
    _assert_other_slots_zero(outputs)


def check_attention_segmented_prefill(
    model_checkpoint: str,
    world_size: int,
    device: str,
    layer_idx: int,
    ut_step: int,
    segment_len: int,
    num_chunks: int,
    block_size: int,
) -> None:
    """Segmented prefill of one layer on one UT step, ``num_chunks`` chunks."""
    total = segment_len * num_chunks
    config = ouro_config(model_checkpoint)
    raw = read_config(model_checkpoint)
    w32, w16 = _weights(model_checkpoint, layer_idx)
    prefix = f"model.layers.{layer_idx}.self_attn"

    torch.manual_seed(42)
    h = torch.randn(total, config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(total, dtype=torch.int64)
    ref32, _ = ref_attention(h.float().unsqueeze(0), positions, w32, prefix, raw)
    ref16, _ = ref_attention(h.unsqueeze(0), positions, w16, prefix, raw)

    num_seq_blocks = total // block_size
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_attention_harness,
            model_checkpoint,
            layer_idx,
            ut_step,
            1 + num_seq_blocks,
            block_size,
            device,
        ),
        device=device,
    )
    cos, sin = _rope(config, positions)
    layer_name = _layer_name(config, ut_step, layer_idx)
    block_table = torch.arange(1, 1 + num_seq_blocks, dtype=torch.int32)
    outputs = None
    for ci in range(num_chunks):
        start, end = ci * segment_len, (ci + 1) * segment_len
        metadata = {
            layer_name: {
                "max_query_len": segment_len,
                "decode_token_threshold": 1,
                "slot_mapping": torch.arange(start, end, dtype=torch.int64)
                + block_size,
                "block_size": block_size,
                "block_table_tensor": block_table.unsqueeze(0),
                "cached_seq_len": torch.tensor([[start]], dtype=torch.int32),
                "kv_segment_size": segment_len,
            }
        }
        executor.dispatch_per_rank(
            [
                (
                    chunk,
                    positions[start:end].to(torch.int32),
                    cos[start:end],
                    sin[start:end],
                    metadata,
                )
                for chunk in h[start:end].chunk(world_size, dim=0)
            ]
        )
        outputs = executor.collect(timeout=COLLECT_TIMEOUT)

    last = (num_chunks - 1) * segment_len
    assert_close_three_way(
        baseline=ref32[0, last:],
        expected=ref16[0, last:],
        actual=torch.cat([o[0].cpu() for o in outputs], dim=0),
        name=f"attn_segmented/chunk{segment_len}x{num_chunks}/tp{world_size}",
    )
    _assert_other_slots_zero(outputs)


def check_mlp(
    model_checkpoint: str,
    world_size: int,
    device: str,
    layer_idx: int,
    num_tokens: int,
    is_prefill: bool,
) -> None:
    """MLP of one layer: SP + reduce-scatter in prefill, all-reduce in decode."""
    config = ouro_config(model_checkpoint)
    w32, w16 = _weights(model_checkpoint, layer_idx)
    prefix = f"model.layers.{layer_idx}.mlp"

    torch.manual_seed(42)
    h = (torch.rand(num_tokens, config.hidden_size) * 2 - 1).to(torch.bfloat16)
    ref32 = ref_mlp(h.float(), w32, prefix)
    ref16 = ref_mlp(h, w16, prefix)

    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_mlp_harness, model_checkpoint, layer_idx, device
        ),
        device=device,
    )
    if is_prefill:
        executor.dispatch_per_rank([(c, True) for c in h.chunk(world_size, dim=0)])
        outputs = executor.collect(timeout=COLLECT_TIMEOUT)
        actual = torch.cat([o.cpu() for o in outputs], dim=0)
    else:
        executor.dispatch(h, False)
        outputs = executor.collect(timeout=COLLECT_TIMEOUT)
        actual = outputs[0].cpu()
    phase = "prefill" if is_prefill else "decode"
    assert_close_three_way(
        baseline=ref32,
        expected=ref16,
        actual=actual,
        name=f"mlp_{phase}/tokens{num_tokens}/tp{world_size}",
    )


def check_attention_decode(
    model_checkpoint: str,
    world_size: int,
    device: str,
    layer_idx: int,
    ut_step: int,
    batch: int,
    context: int,
    blocks_per_seq: int,
    block_size: int,
) -> None:
    """Decode attention of one layer on one UT step, with a cached context.

    The check covers the decode output and the K cache write of the new
    token.
    """
    B, C = batch, context
    config = ouro_config(model_checkpoint)
    raw = read_config(model_checkpoint)
    w32, w16 = _weights(model_checkpoint, layer_idx)
    prefix = f"model.layers.{layer_idx}.self_attn"

    torch.manual_seed(42)
    ctx = torch.randn(B, C, config.hidden_size, dtype=torch.bfloat16)
    h = torch.randn(B, config.hidden_size, dtype=torch.bfloat16)
    ctx_positions = torch.arange(C, dtype=torch.int64)
    dec_positions = torch.tensor([C], dtype=torch.int64)

    refs = {}
    for name, w, dtype in (("fp32", w32, torch.float32), ("bf16", w16, None)):
        outs, new_k, ctx_k, ctx_v = [], [], [], []
        for b in range(B):
            cb = ctx[b : b + 1] if dtype is None else ctx[b : b + 1].to(dtype)
            hb = h[b].view(1, 1, -1) if dtype is None else h[b].view(1, 1, -1).to(dtype)
            _, (kb, vb) = ref_attention(cb, ctx_positions, w, prefix, raw)
            ob, (kn, _) = ref_attention(
                hb, dec_positions, w, prefix, raw, past_kv=(kb, vb)
            )
            outs.append(ob[0, 0])
            new_k.append(kn[0, :, C])
            ctx_k.append(kb[0])
            ctx_v.append(vb[0])
        refs[name] = (
            torch.stack(outs),  # [B, H]
            torch.stack(new_k, dim=1),  # [Nkv, B, D]
            torch.stack(ctx_k),  # [B, Nkv, C, D]
            torch.stack(ctx_v),
        )

    # Block 0 is the null block. Sequence b uses the blocks
    # 1 + blocks_per_seq * b to blocks_per_seq * (b + 1).
    block_table = 1 + torch.arange(B * blocks_per_seq, dtype=torch.int32).view(
        B, blocks_per_seq
    )
    num_blocks = 1 + B * blocks_per_seq
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_attention_harness,
            model_checkpoint,
            layer_idx,
            ut_step,
            num_blocks,
            block_size,
            device,
            context_k=refs["bf16"][2],
            context_v=refs["bf16"][3],
            context_block_table=block_table,
        ),
        device=device,
    )
    positions = torch.full((B,), C, dtype=torch.int32)
    cos, sin = _rope(config, positions)
    slots = (
        block_table[:, C // block_size].to(torch.int64) * block_size + C % block_size
    )
    layer_name = _layer_name(config, ut_step, layer_idx)
    metadata = {
        layer_name: {
            "max_query_len": 1,
            "decode_token_threshold": 1,
            "slot_mapping": slots,
            "block_size": block_size,
            "block_table_tensor": block_table,
            "cached_seq_len": None,
            "kv_segment_size": None,
        }
    }
    executor.dispatch(h, positions, cos, sin, metadata)
    outputs = executor.collect(timeout=COLLECT_TIMEOUT)

    assert_close_three_way(
        baseline=refs["fp32"][0],
        expected=refs["bf16"][0],
        actual=outputs[0][0].cpu().reshape(B, -1),
        name=f"attn_decode/bs{B}/ctx{C}/tp{world_size}",
    )
    assert_close_three_way(
        baseline=refs["fp32"][1],
        expected=refs["bf16"][1],
        actual=gather_cache_tokens([o[1].cpu() for o in outputs], slots, block_size),
        name="attn_decode/k_cache_update",
    )
    _assert_other_slots_zero(outputs)


def crafted_step_hidden(
    gate_weight: torch.Tensor, gate_bias: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Make hidden states whose gate logits are close to ``logits``.

    Args:
        gate_weight: [1, H] gate weight.
        gate_bias: [1] gate bias.
        logits: [N, U] target gate logits.

    Returns:
        [N, U, H] bf16 hidden states.
    """
    N, U = logits.shape
    w = gate_weight[0].float()
    base = torch.randn(N, U, w.shape[0])
    current = base @ w + gate_bias.float()
    shift = (logits - current) / (w @ w)
    return (base + shift.unsqueeze(-1) * w).to(torch.bfloat16)


def check_exit_select(
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    num_ut_steps: int,
    threshold: float | None,
) -> None:
    """The selected hidden state must be bitwise equal to the reference.

    Args:
        gate_weight: [1, H] bf16 checkpoint weight of the early-exit gate.
        gate_bias: [1] bf16 checkpoint bias of the early-exit gate.
        num_ut_steps: Number of UT steps.
        threshold: Exit threshold. None selects the last step.
    """
    select_exit_hidden = src_module("model").select_exit_hidden
    U = num_ut_steps

    torch.manual_seed(42)
    grid = torch.tensor([-6.0, -1.0, 0.0, 1.0, 6.0, 12.0])
    designed = torch.full((U, U), -6.0)
    for u in range(U - 1):
        designed[u, u] = 12.0  # A saturated gate at step u gives an exit at u.
    random = grid[torch.randint(len(grid), (252, U))]
    logits = torch.cat([designed, random], dim=0)
    step_hidden = crafted_step_hidden(gate_weight, gate_bias, logits)
    # Add hidden states with no crafted gate logits.
    plain = torch.randn(64, U, step_hidden.shape[-1]).to(torch.bfloat16)
    step_hidden = torch.cat([step_hidden, plain], dim=0)

    expected, exit_steps = ref_exit_select(
        [step_hidden[:, u].unsqueeze(0) for u in range(U)],
        gate_weight,
        gate_bias,
        threshold,
    )
    actual = select_exit_hidden(
        step_hidden, gate_weight.t().contiguous(), gate_bias, threshold
    )

    exit_steps = exit_steps[0]
    if threshold == 1.0:
        assert set(exit_steps.tolist()) == set(range(U)), (
            f"The inputs do not reach each exit step: {sorted(set(exit_steps.tolist()))}."
        )
    if threshold is None:
        assert torch.all(exit_steps == U - 1)

    mismatch = (actual != expected[0]).any(dim=-1)
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} of {mismatch.numel()} tokens select another step "
        f"than the reference. Reference exit steps of these tokens: "
        f"{exit_steps[mismatch][:16].tolist()}."
    )


def load_gate(model_checkpoint: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the bf16 early-exit gate weight [1, H] and bias [1]."""
    gate = load_tensors(
        model_checkpoint,
        [f"{GATE_PREFIX}.weight", f"{GATE_PREFIX}.bias"],
        torch.bfloat16,
    )
    return gate[f"{GATE_PREFIX}.weight"], gate[f"{GATE_PREFIX}.bias"]


def check_kv_spec_and_binding(model_checkpoint: str, world_size: int) -> None:
    """Check the KV spec, the binding and the cache owners of the full model."""
    config = read_config(model_checkpoint)
    num_kv_layers = config["num_hidden_layers"] * config["total_ut_steps"]
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_kv_spec_harness, model_checkpoint, device="cpu"
        ),
        device="cpu",
    )
    executor.dispatch(torch.zeros(1, dtype=torch.int64))
    outputs = executor.collect(timeout=COLLECT_TIMEOUT)
    assert [int(o[0]) for o in outputs] == [num_kv_layers] * world_size


def check_full_weight_load(model_checkpoint: str, world_size: int) -> None:
    """Load all weights with load_weights, and compare the shards by hand."""
    executor = MPExecutor(
        world_size=world_size,
        model_load=functools.partial(
            load_weight_check_harness, model_checkpoint, device="cpu"
        ),
        device="cpu",
        load_timeout=3600,
    )
    executor.dispatch(torch.zeros(1, dtype=torch.int64))
    outputs = executor.collect(timeout=COLLECT_TIMEOUT)
    # The first and the last layer have 11 checkpoint tensors each, plus 5
    # model-level checkpoint tensors.
    assert [int(o[0]) for o in outputs] == [27] * world_size
