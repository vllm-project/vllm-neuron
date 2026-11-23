# SPDX-License-Identifier: Apache-2.0
"""Kernel and vLLM integration tests for the opt-in NKI stochastic sampler.

Ported from the TrnRL qwen36-trn2 tree (76ca872). The kernel tests run the kernel
under nki.simulate against an FP64 inverse-CDF reference. Set SAMPLER_HARDWARE=1
for kernel hardware tests, or SAMPLER_INTEGRATION_HARDWARE=1 for the serving
bridge hardware tests. STOCHASTIC_SAMPLER_SOURCE points the kernel tests at
another copy of the kernel.
"""

import importlib.util
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nki
import numpy as np
import pytest
import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_neuron.vllm.sample.nki_sampler import (
    NeuronNKISampler,
    has_sampling_filters,
    validate_nki_sampling_params,
)


# Kernel arithmetic and standalone hardware execution.

source = Path(
    os.environ.get(
        "STOCHASTIC_SAMPLER_SOURCE",
        str(
            Path(__file__).parents[4] / "vllm_neuron/functional/stochastic_sampling.py"
        ),
    )
)
spec = importlib.util.spec_from_file_location("stochastic_sampling", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compiled_kernels = {}
artifact_root = Path(
    os.environ.get("SAMPLER_ARTIFACTS") or tempfile.mkdtemp(prefix="sampler-tests-")
)


def validate_inputs(logits, uniforms, temperatures):
    """Validate the standalone host test interface, outside timed device calls."""
    if (
        logits.ndim != 2
        or not 1 <= logits.shape[0] <= 128
        or not 1 <= logits.shape[1] <= 2**24
    ):
        raise ValueError("logits must have shape [1..128, 1..2**24]")
    if logits.dtype.name not in ("float32", "bfloat16"):
        raise ValueError("logits must be float32 or bfloat16")
    expected = (logits.shape[0], 1)
    if uniforms.shape != expected or temperatures.shape != expected:
        raise ValueError("uniforms and temperatures must have shape [batch, 1]")
    if uniforms.dtype != np.float32 or temperatures.dtype != np.float32:
        raise ValueError("uniforms and temperatures must be float32")
    values = logits.astype(np.float32)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("logits cannot contain NaN or positive infinity")
    if not np.isfinite(values).any(axis=1).all():
        raise ValueError("every row needs at least one finite logit")
    if not (np.isfinite(uniforms) & (uniforms >= 0) & (uniforms < 1)).all():
        raise ValueError("uniforms must be finite and in [0, 1)")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inverse = np.reciprocal(temperatures)
    tiny = np.finfo(np.float32).tiny
    if not (
        np.isfinite(temperatures)
        & (temperatures >= tiny)
        & np.isfinite(inverse)
        & (inverse >= tiny)
    ).all():
        raise ValueError(
            "temperatures and their reciprocals must be positive normal FP32 values"
        )


def reference_probabilities(logits, temperatures):
    """FP64 oracle for full-vocabulary temperature-scaled probabilities."""
    values = logits.astype(np.float64)
    scaled = (values - values.max(axis=1, keepdims=True)) / temperatures
    weights = np.exp(scaled)
    return weights / weights.sum(axis=1, keepdims=True)


def reference_sample(logits, uniforms, temperatures):
    """FP64 inverse-CDF reference; a boundary selects the next positive mass."""
    validate_inputs(logits, uniforms, temperatures)
    probabilities = reference_probabilities(logits, temperatures)
    cdf = np.cumsum(probabilities, axis=1)
    selected = (cdf <= uniforms.astype(np.float64)).sum(axis=1)
    last_positive = np.max(
        np.where(probabilities > 0, np.arange(logits.shape[1]), -1), axis=1
    )
    selected = np.minimum(selected, last_positive)
    rows = np.arange(logits.shape[0])
    logprobs = np.log(probabilities[rows, selected])
    return selected.astype(np.int32).reshape(-1, 1), logprobs.astype(
        np.float32
    ).reshape(-1, 1)


def get_compiled(
    logits,
    uniforms,
    temperatures,
    tile_size,
    **kernel_options,
):
    """Test-only cache using the installed SDK's standalone compile API."""
    from nki.compiler.driver import _compile_bir_to_neff
    from nki.compiler.ncc_driver import CompileOptions
    from nki.framework.compiled import compile_kernel_to_nir

    key = (
        logits.shape,
        logits.dtype.name,
        tile_size,
        tuple(sorted(kernel_options.items())),
    )
    if key not in compiled_kernels:
        artifact_root.mkdir(parents=True, exist_ok=True)
        directory = Path(
            tempfile.mkdtemp(
                prefix=f"b{logits.shape[0]}-v{logits.shape[1]}-{logits.dtype.name}-t{tile_size}-",
                dir=artifact_root,
            )
        )
        inputs = {"logits": logits, "uniforms": uniforms, "temperatures": temperatures}
        options = CompileOptions(
            target="trn1",
            lnc=1,
            artifacts_dir=str(directory),
            output_path=str(directory / "kernel.neff"),
        ).disable_backend_optimizations()
        nir = compile_kernel_to_nir(
            module.stochastic_sample_kernel,
            inputs={**inputs, "tile_size": tile_size, **kernel_options},
            compile_opts=options,
        )
        compiled_kernels[key] = _compile_bir_to_neff(nir, options, inputs)
    return compiled_kernels[key]


def hardware_kernel(
    logits,
    uniforms,
    temperatures,
    tile_size,
    **kernel_options,
):
    compiled = get_compiled(logits, uniforms, temperatures, tile_size, **kernel_options)
    arguments = {"logits": logits, "uniforms": uniforms, "temperatures": temperatures}
    inputs = compiled.prepare_inputs(arguments)
    outputs = compiled.prepare_outputs()
    compiled.execute(inputs, outputs)
    return tuple(outputs.values())


kernel = (
    hardware_kernel
    if os.environ.get("SAMPLER_HARDWARE") == "1"
    else nki.simulate(module.stochastic_sample_kernel)
)


def run(logits, uniforms, temperatures=None, tile_size=4096, **kernel_options):
    logits = np.asarray(logits, dtype=np.float32)
    uniforms = np.asarray(uniforms, dtype=np.float32).reshape(-1, 1)
    if temperatures is None:
        temperatures = np.ones_like(uniforms)
    temperatures = np.asarray(temperatures, dtype=np.float32).reshape(-1, 1)
    validate_inputs(logits, uniforms, temperatures)
    return kernel(logits, uniforms, temperatures, tile_size=tile_size, **kernel_options)


@pytest.mark.parametrize("width", [1, 2, 31, 127, 129, 256, 4095, 4096, 4097, 8193])
def test_uniform_distribution_exact_boundaries_and_partial_tiles(width):
    uniforms = np.array(
        [0.0, 0.125, 0.25, 0.5, 0.75, 0.9, np.nextafter(np.float32(1), np.float32(0))],
        np.float32,
    )
    logits = np.zeros((len(uniforms), width), np.float32)
    ids, scores = run(logits, uniforms)
    expected = np.minimum((uniforms.astype(np.float64) * width).astype(int), width - 1)
    np.testing.assert_array_equal(ids[:, 0], expected)
    np.testing.assert_allclose(scores[:, 0], -np.log(width), atol=3e-5)


def test_masked_tokens_including_trailing_and_all_masked_tiles():
    logits = np.full((4, 8201), -np.inf, np.float32)
    logits[0, 8199] = 0
    logits[1, 0] = 0
    logits[2, [1, 4096, 8192]] = 0
    logits[3, [2, 4097]] = 0
    ids, scores = run(logits, [0, np.nextafter(np.float32(1), np.float32(0)), 0.5, 0.5])
    np.testing.assert_array_equal(ids[:, 0], [8199, 0, 4096, 4097])
    np.testing.assert_allclose(scores[:, 0], [0, 0, -np.log(3), -np.log(2)], atol=3e-5)


def test_temperature_and_selected_logprob_match_reference():
    rng = np.random.default_rng(827)
    logits = (rng.normal(size=(8, 4101)) * 3).astype(np.float32)
    uniforms = rng.uniform(0.1, 0.9, (8, 1)).astype(np.float32)
    temps = np.array([0.1, 0.3, 0.7, 1, 1.5, 2, 4, 10], np.float32).reshape(-1, 1)
    expected_ids, expected_scores = reference_sample(logits, uniforms, temps)
    ids, scores = run(logits, uniforms, temps)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(scores, expected_scores, atol=5e-5, rtol=2e-5)


@pytest.mark.parametrize("width", [4101, 248320])
def test_raw_and_processed_scores_preserve_sampling(width):
    rng = np.random.default_rng(218)
    logits = (rng.normal(size=(4, width)) * 3).astype(np.float32)
    logits[0] += 10000  # Both normalizers must tolerate a common shift.
    logits[1, :4096] = -np.inf
    temps = np.array([0.3, 1.0, 2.0, 10.0], np.float32).reshape(-1, 1)
    uniforms = rng.uniform(0.1, 0.9, (4, 1)).astype(np.float32)
    processed = run(logits, uniforms, temps, return_ranks=True)
    raw = run(logits, uniforms, temps, return_ranks=True, raw_logprobs=True)
    np.testing.assert_array_equal(raw[0], processed[0])
    np.testing.assert_array_equal(raw[2], processed[2])
    rows, selected = np.arange(4), raw[0][:, 0]
    for outputs, score_temps in ((raw, np.ones_like(temps)), (processed, temps)):
        probs = reference_probabilities(logits, score_temps)
        np.testing.assert_allclose(
            outputs[1][:, 0], np.log(probs[rows, selected]), atol=5e-5, rtol=2e-5
        )
        np.testing.assert_array_equal(
            outputs[2][:, 0], (logits >= logits[rows, selected, None]).sum(-1)
        )
    np.testing.assert_allclose(raw[1][1], processed[1][1], atol=5e-5)
    assert not np.allclose(raw[1], processed[1])


def test_raw_logprob_stays_finite_when_raw_weight_underflows():
    logits = np.full((1, 4101), -np.inf, np.float32)
    logits[0, [0, 4096, 4100]] = [0.0, -1000.0, -1001.0]
    ids, scores, ranks = run(
        logits, [0.7], [1000.0], raw_logprobs=True, return_ranks=True
    )
    np.testing.assert_array_equal(ids, [[4096]])
    np.testing.assert_allclose(scores, [[-1000.0]], atol=1e-4)
    np.testing.assert_array_equal(ranks, [[2]])


@pytest.mark.parametrize("raw", [False, True])
def test_rank_does_not_merge_nearby_logits_after_centering(raw):
    logits = np.full((1, 4101), -np.inf, np.float32)
    logits[0, [0, 4096, 4100]] = [
        100.0,
        1.0,
        np.nextafter(np.float32(1), np.float32(2)),
    ]
    ids, _, ranks = run(logits, [0.9], [100.0], raw_logprobs=raw, return_ranks=True)
    np.testing.assert_array_equal(ids, [[4100]])
    np.testing.assert_array_equal(ranks, [[2]])


def test_request_reordering_preserves_decisions():
    rng = np.random.default_rng(172)
    logits = rng.normal(size=(8, 129)).astype(np.float32)
    uniforms = rng.random((8, 1), dtype=np.float32)
    temperatures = rng.uniform(0.4, 2, (8, 1)).astype(np.float32)
    order = np.array([4, 2, 7, 0, 1, 5, 3, 6])
    ids, scores = run(logits, uniforms, temperatures)
    shuffled_ids, shuffled_scores = run(
        logits[order], uniforms[order], temperatures[order]
    )
    np.testing.assert_array_equal(shuffled_ids, ids[order])
    np.testing.assert_array_equal(shuffled_scores, scores[order])


def test_no_256_candidate_cap():
    width = 248320
    logits = np.full((1, width), -np.inf, np.float32)
    logits[0, width - 1] = 100
    ids, scores = run(logits, [0.8])
    np.testing.assert_array_equal(ids, [[width - 1]])
    np.testing.assert_allclose(scores, 0, atol=3e-5)


@pytest.mark.parametrize("batch", [1, 8])
def test_bfloat16_logits(batch):
    import ml_dtypes

    rng = np.random.default_rng(429)
    logits = rng.normal(size=(batch, 1031)).astype(ml_dtypes.bfloat16)
    uniforms = rng.random((batch, 1), dtype=np.float32)
    temperatures = rng.uniform(0.5, 2, (batch, 1)).astype(np.float32)
    validate_inputs(logits, uniforms, temperatures)
    ids, scores = kernel(logits, uniforms, temperatures, tile_size=512)
    reference_ids, reference_scores = reference_sample(logits, uniforms, temperatures)
    np.testing.assert_array_equal(ids, reference_ids)
    np.testing.assert_allclose(scores, reference_scores, atol=5e-5, rtol=2e-5)


def test_small_temperature_and_large_common_logit_shift():
    logits = np.array([[10000, 10001, -10000], [0, -100, -1000]], np.float32)
    ids, scores = run(logits, [0.2, 0.99], [1e-5, 1])
    np.testing.assert_array_equal(ids[:, 0], [1, 0])
    np.testing.assert_allclose(scores, 0, atol=3e-5)


def test_full_vocabulary_random_inputs_and_logprobs():
    rng = np.random.default_rng(6008)
    logits = rng.normal(size=(8, 248320)).astype(np.float32)
    uniforms = rng.random((8, 1), dtype=np.float32)
    temperatures = np.ones((8, 1), np.float32)
    ids, scores = run(logits, uniforms, temperatures)
    values = logits.astype(np.float64)
    weights = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probabilities, axis=1)
    rows = np.arange(len(ids))
    chosen = ids[:, 0]
    lower = np.where(chosen > 0, cdf[rows, np.maximum(0, chosen - 1)], 0)
    upper = cdf[rows, chosen]
    # Float32 tile scans can differ at a CDF boundary by a few ulps.
    assert np.all(uniforms[:, 0] >= lower - 2e-6)
    assert np.all(uniforms[:, 0] <= upper + 2e-6)
    np.testing.assert_allclose(
        scores[:, 0], np.log(probabilities[rows, chosen]), atol=5e-5, rtol=2e-5
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("uniform", -0.1),
        ("uniform", 1.0),
        ("uniform", np.nan),
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("temperature", np.inf),
        ("temperature", 1e-40),
        ("temperature", np.finfo(np.float32).max),
        ("logit", np.nan),
        ("logit", np.inf),
    ],
)
def test_invalid_values_rejected(field, value):
    logits = np.zeros((1, 2), np.float32)
    uniforms = np.array([[0.5]], np.float32)
    temperatures = np.ones((1, 1), np.float32)
    {"uniform": uniforms, "temperature": temperatures, "logit": logits}[field][0, 0] = (
        value
    )
    with pytest.raises(ValueError):
        validate_inputs(logits, uniforms, temperatures)


def test_all_masked_row_rejected():
    with pytest.raises(ValueError, match="finite logit"):
        validate_inputs(
            np.full((1, 8), -np.inf, np.float32),
            np.array([[0.5]], np.float32),
            np.ones((1, 1), np.float32),
        )


@pytest.mark.parametrize("tile_size", [2048, 4096, 8192])
def test_vocabulary_partition_layouts(tile_size):
    rng = np.random.default_rng(738)
    logits = rng.normal(size=(4, 248320)).astype(np.float32)
    logits[0] = 0
    logits[1] = -np.inf
    logits[1, [0, 248319]] = [0, 1]
    logits[2, :240000] = -np.inf
    logits[3] *= 5
    uniforms = np.array(
        [0.5, 0.99, np.nextafter(np.float32(1), np.float32(0)), 0.31], np.float32
    ).reshape(-1, 1)
    temperatures = np.array([1, 0.7, 2, 0.3], np.float32).reshape(-1, 1)
    ids, scores = kernel(logits, uniforms, temperatures, tile_size=tile_size)
    probs = reference_probabilities(logits, temperatures)
    chosen = probs[np.arange(4), ids[:, 0]]
    assert (chosen > 0).all()
    cdf = probs.cumsum(axis=1)
    upper = cdf[np.arange(4), ids[:, 0]]
    lower = upper - chosen
    assert (uniforms[:, 0] >= lower - 2e-6).all()
    assert (uniforms[:, 0] <= upper + 2e-6).all()
    np.testing.assert_allclose(scores[:, 0], np.log(chosen), atol=5e-5, rtol=2e-5)


def test_packed_empirical_distribution():
    # Exercise three packed partitions, complementing single-tile checks.
    probs = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05]
    )
    batch, draws = 8, 1024
    logits = np.tile(np.log(probs), (batch, 1)).astype(np.float32)
    temperatures = np.ones((batch, 1), np.float32)
    rng = np.random.default_rng(7342)
    counts = np.zeros(len(probs), np.int64)
    for _ in range(draws // batch):
        ids, scores = kernel(
            logits, rng.random((batch, 1), dtype=np.float32), temperatures, tile_size=4
        )
        counts += np.bincount(ids[:, 0], minlength=len(probs))
        np.testing.assert_allclose(scores[:, 0], np.log(probs[ids[:, 0]]), atol=5e-5)
    assert (
        np.abs(counts - draws * probs) <= 5 * np.sqrt(draws * probs * (1 - probs)) + 1
    ).all()


@pytest.mark.parametrize(
    "vocab,tile_size,message",
    [
        (8, 0, "tile_size"),
        (8, 8193, "tile_size"),
        # NKI rejects zero-sized tensors before entering the kernel.
        (0, 4096, "positive integers"),
        (248320, 1024, "1..128"),
    ],
)
def test_layout_guards(vocab, tile_size, message):
    # These guards execute before device work; check them through the simulator.
    with pytest.raises(AssertionError, match=message):
        nki.simulate(module.stochastic_sample_kernel)(
            np.zeros((1, vocab), np.float32),
            np.array([[0.5]], np.float32),
            np.ones((1, 1), np.float32),
            tile_size=tile_size,
        )


def test_single_tile_empirical():
    # Exercise one tile directly.
    batch, draws = 8, 1024
    logits = np.tile(np.array([-2, -1, 0, 0.5, 1, 2], np.float32), (batch, 1))
    temperatures = np.full((batch, 1), 1.5, np.float32)
    expected = reference_probabilities(logits, temperatures)[0]
    rng = np.random.default_rng(7462)
    counts = np.zeros(logits.shape[1], np.int64)
    for _ in range(draws // batch):
        ids, scores = kernel(
            logits,
            rng.random((batch, 1), dtype=np.float32),
            temperatures,
            tile_size=8,
        )
        counts += np.bincount(ids[:, 0], minlength=len(counts))
        np.testing.assert_allclose(scores[:, 0], np.log(expected[ids[:, 0]]), atol=5e-5)
    assert (counts[expected == 0] == 0).all()
    assert (
        np.abs(counts - draws * expected)
        <= 5 * np.sqrt(draws * expected * (1 - expected)) + 1
    ).all()


# vLLM adapter, routing, request RNG and serving bridge.


def metadata(temperatures, k=None, p=None, count=0):
    size = len(temperatures)
    return SamplingMetadata(
        temperature=torch.tensor(temperatures, dtype=torch.float32),
        all_greedy=all(t == 0 for t in temperatures),
        all_random=all(t > 0 for t in temperatures),
        top_k=None if k is None else torch.tensor(k),
        top_p=None if p is None else torch.tensor(p),
        generators={i: torch.Generator().manual_seed(417 + i) for i in range(size)},
        max_num_logprobs=count,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(size),
        presence_penalties=torch.zeros(size),
        repetition_penalties=torch.ones(size),
        output_token_ids=[[] for _ in range(size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(argmax_invariant=[], non_argmax_invariant=[]),
    )


class TestNKIIntegration:
    """Keep eager execution scoped to the integration cases."""

    @pytest.fixture(autouse=True)
    def eager(self):
        with torch.compiler.set_stance("force_eager"):
            yield

    @staticmethod
    def _runner(md, sampler, cpu_sampler, vocab_size, num_reqs=2):
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        runner = SimpleNamespace(
            input_batch=SimpleNamespace(sampling_metadata=md, num_reqs=num_reqs),
            sampler=sampler,
            cpu_sampler=cpu_sampler,
            vocab_size=vocab_size,
            nki_sampling=True,
            nki_in_graph_sampling=False,
            on_device_sampling=False,
            use_async_scheduling=False,
            _on_device_nki_scores=None,
        )
        runner._sample_separate_nki = (
            lambda logits,
            metadata: runner_module.NeuronModelRunner._sample_separate_nki(
                runner, logits, metadata
            )
        )
        return runner

    @pytest.mark.parametrize("temperatures", [[0.7, 1.0], [0.0, 0.7]])
    @pytest.mark.parametrize("filters", ["k", "p", "both", "mixed"])
    @pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
    def test_filtered_batch_reuses_upstream_cpu_sampler(
        self, temperatures, filters, mode
    ):
        """A filtered batch keeps upstream decisions, scores and RNG on the CPU."""
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        logits = torch.randn(3, 32, generator=torch.Generator().manual_seed(42))
        original = logits.clone()
        k = (
            [32, 16]
            if filters == "mixed"
            else [8, 16]
            if filters in ("k", "both")
            else None
        )
        p = [0.9, 0.95] if filters in ("p", "both") else None
        md = metadata(temperatures, k=k, p=p)
        reference_md = metadata(temperatures, k=k, p=p)
        sampler = Sampler(logprobs_mode=mode)
        nki = Mock(side_effect=AssertionError("Filtered batch reached NKI"))
        runner = self._runner(md, nki, sampler, 32)
        group = SimpleNamespace(rank_in_group=0, world_size=1)
        with patch.object(runner_module, "get_tp_group", return_value=group):
            output = runner_module.NeuronModelRunner._sample(runner, logits)
        reference = sampler(logits[:2].clone(), reference_md)
        assert torch.equal(output.sampled_token_ids, reference.sampled_token_ids)
        torch.testing.assert_close(
            output.logprobs_tensors.logprobs, reference.logprobs_tensors.logprobs
        )
        assert torch.equal(logits, original)
        for row in range(2):
            assert torch.equal(
                md.generators[row].get_state(), reference_md.generators[row].get_state()
            )
        nki.assert_not_called()

    @pytest.mark.parametrize("k", [None, [-1, 0], [32, 32]])
    def test_runner_keeps_unfiltered_batch_on_nki(self, k):
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        logits = torch.randn(3, 32)
        md = metadata([0.7, 1.0], k=k, p=[1.0, 1.0])
        nki = Mock(return_value="nki-result")
        cpu = Mock(side_effect=AssertionError("Unfiltered batch reached CPU"))
        runner = self._runner(md, nki, cpu, 32)
        group = SimpleNamespace(rank_in_group=0, world_size=1)
        with patch.object(runner_module, "get_tp_group", return_value=group):
            assert (
                runner_module.NeuronModelRunner._sample(runner, logits) == "nki-result"
            )
        cpu.assert_not_called()
        assert nki.call_args.kwargs["sampling_metadata"] is md
        assert nki.call_args.kwargs["logits"].data_ptr() == logits.data_ptr()
        assert nki.call_args.kwargs["logits"].shape == (2, 32)

    @pytest.mark.parametrize(
        "k,p,expected",
        [
            (None, None, False),
            ([32, 32], [1.0, 1.0], False),
            ([-1, 0], None, False),
            ([32, 4], None, True),
            (None, [1.0, 0.95], True),
        ],
    )
    def test_filter_routing(self, k, p, expected):
        assert has_sampling_filters(metadata([0.7, 1.0], k=k, p=p), 32) is expected

    def test_tp_only_leader_samples_and_broadcasts_failure(self):
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        sampler = Mock(return_value="authoritative")
        runner = self._runner(metadata([0.7, 1.0]), sampler, sampler, 100)
        leader = SimpleNamespace(
            rank_in_group=0,
            world_size=8,
            broadcast_object=Mock(side_effect=lambda x, src: x),
        )
        with patch.object(runner_module, "get_tp_group", return_value=leader):
            result = runner_module.NeuronModelRunner._sample(
                runner, torch.zeros(2, 100)
            )
        assert result == "authoritative"
        sampler.assert_called_once()
        follower = SimpleNamespace(
            rank_in_group=3,
            world_size=8,
            broadcast_object=Mock(return_value=("authoritative", None)),
        )
        with patch.object(runner_module, "get_tp_group", return_value=follower):
            assert (
                runner_module.NeuronModelRunner._sample(runner, torch.zeros(2, 100))
                == result
            )
        sampler.assert_called_once()
        with patch.object(runner_module, "get_tp_group", return_value=leader):
            with pytest.raises(RuntimeError, match="full vocabulary width"):
                runner_module.NeuronModelRunner._sample(runner, torch.zeros(2, 10))
            assert leader.broadcast_object.call_args.args[0][0] is None
            sampler.side_effect = ValueError("sampling error")
            with pytest.raises(RuntimeError, match="sampling error"):
                runner_module.NeuronModelRunner._sample(runner, torch.zeros(2, 100))
        follower.broadcast_object.return_value = leader.broadcast_object.call_args.args[
            0
        ]
        with patch.object(runner_module, "get_tp_group", return_value=follower):
            with pytest.raises(RuntimeError, match="sampling error"):
                runner_module.NeuronModelRunner._sample(runner, torch.zeros(2, 100))
        assert sampler.call_count == 2

    @pytest.mark.parametrize(
        "field,value",
        [
            ("min_p", 0.1),
            ("min_tokens", 4),
            ("logit_bias", {1: 2}),
            ("thinking_token_budget", 10),
            ("thinking_token_budget", 0),
            ("prompt_logprobs", 0),
        ],
    )
    def test_unwired_processor_requests_fail_explicitly(self, field, value):
        params = SimpleNamespace(prompt_logprobs=None, logprobs=None)
        setattr(params, field, value)
        with pytest.raises(ValueError, match=field):
            validate_nki_sampling_params(params)

    @pytest.fixture(params=["raw_logprobs", "processed_logprobs"])
    def nki_sampler(self, monkeypatch, request):
        """Run the actual device arithmetic while testing the vLLM host contract."""
        import nki
        import numpy as np

        from vllm_neuron.functional.stochastic_sampling import stochastic_sample_kernel

        sampler = NeuronNKISampler(logprobs_mode=request.param)

        def run(logits, uniforms, temperatures, greedy):
            def array(value):
                return (
                    None
                    if value is None
                    else value.float().numpy()
                    if value.dtype == torch.bfloat16
                    else value.numpy()
                )

            outputs = nki.simulate(stochastic_sample_kernel)(
                array(logits),
                array(uniforms),
                array(temperatures),
                greedy_rows=array(greedy),
                return_ranks=True,
                raw_logprobs=sampler.logprobs_mode == "raw_logprobs",
            )
            return tuple(torch.from_numpy(np.asarray(value)) for value in outputs)

        monkeypatch.setattr(sampler, "_run_kernel", run)
        return sampler

    def test_warmup_preserves_rng_and_reuses_shapes(self, nki_sampler):
        logits = torch.randn(3, 32, generator=torch.Generator().manual_seed(71))
        original = logits.clone()
        md = metadata([0.0, 0.7, 1.0])
        rng = torch.random.get_rng_state()
        request_rng = {i: gen.get_state() for i, gen in md.generators.items()}
        with patch.object(
            nki_sampler, "_run_kernel", wraps=nki_sampler._run_kernel
        ) as run:
            nki_sampler.warmup(logits, 3)
            assert [call.args[0].shape[0] for call in run.call_args_list] == [1, 2, 3]
            nki_sampler.warmup(logits[:1], 3)
            assert run.call_count == 3
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert torch.equal(logits, original)
        for row, gen in md.generators.items():
            assert torch.equal(gen.get_state(), request_rng[row])
        output = nki_sampler(logits, md)
        reference = nki_sampler(logits, metadata([0.0, 0.7, 1.0]))
        assert torch.equal(output.sampled_token_ids, reference.sampled_token_ids)
        torch.testing.assert_close(
            output.logprobs_tensors.logprobs, reference.logprobs_tensors.logprobs
        )

    def test_warmup_failure_is_not_marked_ready(self):
        sampler = NeuronNKISampler()
        logits = torch.zeros(1, 32)
        with patch.object(
            sampler, "_run_kernel", side_effect=RuntimeError("compile failed")
        ):
            with pytest.raises(RuntimeError, match="compile failed"):
                sampler.warmup(logits, 2)
        with patch.object(sampler, "_run_kernel", return_value=()) as run:
            sampler.warmup(logits, 2)
            assert run.call_count == 2

    def test_nki_mixed_rows_scores_ranks_and_rng(self, nki_sampler):
        logits = torch.tensor([[2.0, 2.0, 1.0, -float("inf")], [0.0, 1.0, 2.0, 3.0]])
        md = metadata([0.0, 0.7])
        states = {i: gen.get_state() for i, gen in md.generators.items()}
        expected_gen = torch.Generator()
        expected_gen.set_state(states[1])
        torch.rand((), generator=expected_gen)
        output = nki_sampler(logits, md)
        expected = logits.clone()
        if nki_sampler.logprobs_mode == "processed_logprobs":
            expected[1:] /= 0.7
        expected = expected.log_softmax(-1)
        selected = output.sampled_token_ids.long()
        scores = expected.gather(1, selected)
        assert selected[0].item() == 0
        torch.testing.assert_close(
            output.logprobs_tensors.logprobs, scores, atol=5e-5, rtol=2e-5
        )
        assert torch.equal(
            output.logprobs_tensors.selected_token_ranks, (expected >= scores).sum(-1)
        )
        assert torch.equal(md.generators[0].get_state(), states[0])
        assert torch.equal(md.generators[1].get_state(), expected_gen.get_state())

    def test_nki_request_rng_reordering_and_slot_reuse(self, nki_sampler):
        logits = torch.randn(3, 17, generator=torch.Generator().manual_seed(42))
        a = metadata([0.7, 0.0, 1.5])
        b = metadata([1.5, 0.7, 0.0])
        order = torch.tensor([2, 0, 1])
        b.generators = {
            i: torch.Generator().manual_seed(417 + row)
            for i, row in enumerate(order.tolist())
        }
        for _ in range(3):
            x = nki_sampler(logits, a)
            y = nki_sampler(logits[order], b)
            assert torch.equal(x.sampled_token_ids[order], y.sampled_token_ids)
            torch.testing.assert_close(
                x.logprobs_tensors.logprobs[order], y.logprobs_tensors.logprobs
            )
        # A new request in an old slot starts from its own seed.
        a.generators[0] = torch.Generator().manual_seed(999)
        fresh = metadata([0.7])
        fresh.generators[0] = torch.Generator().manual_seed(999)
        assert torch.equal(
            nki_sampler(logits, a).sampled_token_ids[:1],
            nki_sampler(logits[:1], fresh).sampled_token_ids,
        )

    @pytest.mark.parametrize(
        "changes",
        [
            {"presence_penalty": 0.2},
            {"logprobs": 1},
            {"prompt_logprobs": 0},
            {"logit_bias": {1: 1.0}},
            {"allowed_token_ids": [1]},
            {"min_p": 0.1},
        ],
    )
    def test_nki_rejects_requests_before_dispatch(self, changes):
        from vllm import SamplingParams
        from vllm_neuron.vllm.platform import NeuronPlatform

        sampler = NeuronNKISampler()
        with (
            patch.multiple(
                NeuronPlatform,
                _nki_sampling=True,
                _nki_in_graph=False,
                _sampling_logprobs_mode="raw_logprobs",
            ),
            patch.object(sampler, "_run_kernel") as run,
        ):
            with pytest.raises(ValueError):
                NeuronPlatform.validate_request({}, SamplingParams(**changes))
            run.assert_not_called()

    @pytest.mark.parametrize("k", [None, [-1], [0], [8]])
    def test_nki_accepts_disabled_filter_metadata(self, nki_sampler, k):
        md = metadata([1.0], k=k, p=[1.0], count=None)
        with patch.object(
            nki_sampler, "_run_kernel", wraps=nki_sampler._run_kernel
        ) as run:
            output = nki_sampler(torch.zeros(1, 8), md)
        assert len(run.call_args.args) == 4  # logits, uniform, temperature, greedy
        assert output.logprobs_tensors is None

    @pytest.mark.parametrize("k,p", [([2], None), (None, [0.95])])
    @pytest.mark.parametrize("temperature", [0.0, 0.7])
    def test_nki_filters_rejected_before_rng_or_dispatch(self, k, p, temperature):
        sampler = NeuronNKISampler()
        md = metadata([temperature], k=k, p=p)
        state = md.generators[0].get_state()
        with patch.object(sampler, "_run_kernel") as run:
            with pytest.raises(AssertionError, match="Filtered batches"):
                sampler(torch.zeros(1, 8), md)
            run.assert_not_called()
        assert torch.equal(md.generators[0].get_state(), state)

    def test_nki_frontend_and_config_guards(self):
        from vllm import SamplingParams

        from vllm_neuron.model.neuron_config import NeuronConfig
        from vllm_neuron.vllm.platform import NeuronPlatform

        # The dataclass parses configuration; the platform owns compatibility.
        assert NeuronConfig.from_dict({}).stochastic_sampling_backend == "cpu"
        assert (
            NeuronConfig.from_dict(
                {"stochastic_sampling_backend": "nki"}
            ).stochastic_sampling_backend
            == "nki"
        )
        config = NeuronConfig.from_dict(
            {
                "stochastic_sampling_backend": "nki",
                "on_device_sampling_config": None,
            }
        )
        assert config.stochastic_sampling_backend == "nki"
        with pytest.raises(ValueError, match="stochastic_sampling_backend"):
            NeuronConfig.from_dict({"stochastic_sampling_backend": "gpu"})
        with patch.multiple(
            NeuronPlatform,
            _nki_sampling=True,
            _nki_in_graph=False,
            _sampling_logprobs_mode="raw_logprobs",
        ):
            NeuronPlatform.validate_request({}, SamplingParams(logprobs=0))
            # The separate sampler sends a filtered batch to the CPU sampler.
            for params in (SamplingParams(top_k=20), SamplingParams(top_p=0.95)):
                NeuronPlatform.validate_request({}, params)
            for params in (
                SamplingParams(logprobs=2),
                SamplingParams(presence_penalty=0.2),
            ):
                with pytest.raises(ValueError):
                    NeuronPlatform.validate_request({}, params)

    def test_the_cpu_backend_validates_nothing_new(self):
        from vllm import SamplingParams
        from vllm_neuron.vllm.platform import NeuronPlatform

        with patch.multiple(NeuronPlatform, _nki_sampling=False, _nki_in_graph=False):
            for params in (
                SamplingParams(logprobs=5),
                SamplingParams(top_k=20),
                SamplingParams(prompt_logprobs=1),
                SamplingParams(min_p=0.1),
            ):
                NeuronPlatform.validate_request({}, params)

    @pytest.mark.parametrize(
        "backend,embedded,async_,spec,mode,processors,error",
        [
            (
                "bad",
                False,
                False,
                False,
                "raw_logprobs",
                None,
                "stochastic_sampling_backend",
            ),
            # In-graph NKI sampling (embedded config + nki backend), sync or async.
            ("nki", True, False, False, "raw_logprobs", None, None),
            ("nki", True, True, False, "processed_logprobs", None, None),
            ("nki", True, False, True, "raw_logprobs", None, "speculative"),
            ("nki", False, True, False, "raw_logprobs", None, "async scheduling"),
            ("nki", False, False, True, "raw_logprobs", None, "speculative"),
            (
                "nki",
                False,
                False,
                False,
                "raw_logprobs",
                ["custom"],
                "custom logits processors",
            ),
            ("nki", True, True, False, "raw_logits", None, "logprobs_mode"),
            ("nki", False, False, False, "raw_logprobs", None, None),
            # The default backend keeps the configuration of main.
            ("cpu", False, True, False, "raw_logprobs", None, None),
            ("cpu", True, False, False, "processed_logprobs", None, None),
            ("cpu", True, True, True, "raw_logits", ["custom"], None),
        ],
    )
    def test_platform_owns_sampler_configuration(
        self,
        backend,
        embedded,
        async_,
        spec,
        mode,
        processors,
        error,
    ):
        from vllm_neuron.vllm.platform import NeuronPlatform

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                logprobs_mode=mode, logits_processors=processors
            ),
            scheduler_config=SimpleNamespace(async_scheduling=async_),
            speculative_config=object() if spec else None,
        )
        neuron_config = {
            "stochastic_sampling_backend": backend,
            "on_device_sampling_config": {} if embedded else None,
        }
        with patch.multiple(
            NeuronPlatform,
            _nki_sampling=False,
            _nki_in_graph=False,
            _sampling_logprobs_mode="raw_logprobs",
        ):
            if error:
                with pytest.raises(ValueError, match=error):
                    NeuronPlatform._configure_nki_sampling(config, neuron_config)
            else:
                NeuronPlatform._configure_nki_sampling(config, neuron_config)
                assert NeuronPlatform._nki_sampling is (backend == "nki")
                assert NeuronPlatform._nki_in_graph is (backend == "nki" and embedded)

    def test_in_graph_nki_allows_async_scores_and_rejects_filters(self):
        from vllm import SamplingParams
        from vllm_neuron.vllm.platform import NeuronPlatform

        with patch.multiple(
            NeuronPlatform,
            _nki_sampling=True,
            _nki_in_graph=True,
            _sampling_logprobs_mode="processed_logprobs",
        ):
            NeuronPlatform.validate_request({}, SamplingParams(logprobs=0))
            # SkyRL's request: logprobs, a seed, top_k -1 and top_p 1.0.
            NeuronPlatform.validate_request(
                {},
                SamplingParams(
                    temperature=0.7, top_k=-1, top_p=1.0, logprobs=0, seed=1234
                ),
            )
            for params in (SamplingParams(top_k=20), SamplingParams(top_p=0.9)):
                with pytest.raises(ValueError, match="top_k/top_p"):
                    NeuronPlatform.validate_request({}, params)

    def test_in_graph_nki_params_match_separate_sampler_rng(self):
        from types import SimpleNamespace

        import numpy as np
        from vllm_neuron.vllm.worker.neuron_model_runner import (
            build_nki_sampling_params_tensor,
        )

        def batch():
            return SimpleNamespace(
                num_reqs=3,
                temperature_cpu=np.array([0.0, 0.7, 1.3, 9.0], dtype=np.float32),
                generators={
                    1: torch.Generator().manual_seed(5),
                    2: torch.Generator().manual_seed(6),
                },
            )

        b = batch()
        params = build_nki_sampling_params_tensor(b, 5)
        ref = batch()
        expected_u1 = torch.rand((), generator=ref.generators[1])
        expected_u2 = torch.rand((), generator=ref.generators[2])
        assert params.shape == (5, 3)
        # Greedy row: no RNG, temperature 1, greedy flag.
        assert params[0].tolist() == [0.0, 1.0, 1.0]
        assert params[1, 0] == expected_u1 and params[2, 0] == expected_u2
        assert torch.allclose(params[1:3, 1], torch.tensor([0.7, 1.3]))
        assert params[1:3, 2].tolist() == [0.0, 0.0]
        # Rows beyond num_reqs are greedy padding.
        assert params[3:].tolist() == [[0.0, 1.0, 1.0]] * 2
        # Exactly one draw per stochastic row.
        assert torch.rand((), generator=b.generators[1]) == torch.rand(
            (), generator=ref.generators[1]
        )

    def test_in_graph_nki_logprobs_trim_padding(self):
        from vllm_neuron.vllm.worker.neuron_model_runner import _nki_logprobs_tensors

        ids = torch.tensor([7, 9, 0], dtype=torch.int32)
        scores = torch.tensor([[-0.5], [-1.5], [0.0]])
        ranks = torch.tensor([[1], [3], [1]], dtype=torch.int32)
        out = _nki_logprobs_tensors(ids, scores, ranks, 2)
        assert out.logprob_token_ids.tolist() == [[7], [9]]
        assert out.logprobs.tolist() == [[-0.5], [-1.5]]
        assert out.selected_token_ranks.tolist() == [1, 3]
        with pytest.raises(ValueError, match="non-finite"):
            _nki_logprobs_tensors(
                ids, torch.tensor([[float("nan")], [0.0], [0.0]]), ranks, 2
            )

    def test_nki_score_mode_rejected_at_frontend(self):
        from vllm import SamplingParams
        from vllm_neuron.vllm.platform import NeuronPlatform

        with patch.multiple(
            NeuronPlatform,
            _nki_sampling=True,
            _nki_in_graph=False,
            _sampling_logprobs_mode="raw_logits",
        ):
            NeuronPlatform.validate_request({}, SamplingParams())
            with pytest.raises(ValueError, match="score modes"):
                NeuronPlatform.validate_request({}, SamplingParams(logprobs=0))

    def test_cpu_preparation_preserves_rng_and_original_logits(self):
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        logits = torch.randn(2, 11)
        before = logits.clone()
        md = metadata([0.7, 1.0], k=[3, 5])

        def sample(logits, sampling_metadata):
            assert sampling_metadata is not md
            assert sampling_metadata.generators is md.generators
            assert sampling_metadata.output_token_ids is md.output_token_ids
            assert sampling_metadata.temperature.device.type == "cpu"
            assert sampling_metadata.top_k.device.type == "cpu"
            logits.zero_()
            return "cpu-result"

        runner = self._runner(
            md,
            Mock(side_effect=AssertionError("Filtered batch reached NKI")),
            Mock(side_effect=sample),
            11,
        )
        group = SimpleNamespace(rank_in_group=0, world_size=1)
        with patch.object(runner_module, "get_tp_group", return_value=group):
            assert (
                runner_module.NeuronModelRunner._sample(runner, logits) == "cpu-result"
            )
        torch.testing.assert_close(logits, before)

    def test_follower_does_not_prepare_metadata(self):
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        runner = self._runner(metadata([0.7, 1.0], k=[3, 5]), Mock(), Mock(), 11)
        group = SimpleNamespace(
            rank_in_group=1,
            world_size=8,
            broadcast_object=Mock(return_value=("cpu-result", None)),
        )
        with (
            patch.object(runner_module, "get_tp_group", return_value=group),
            patch.object(
                runner_module,
                "replace",
                side_effect=AssertionError("Follower prepared metadata"),
            ),
        ):
            assert (
                runner_module.NeuronModelRunner._sample(runner, torch.zeros(2, 11))
                == "cpu-result"
            )
        runner.cpu_sampler.assert_not_called()
        runner.sampler.assert_not_called()

    @pytest.mark.parametrize("wants_logprobs", [False, True])
    def test_in_graph_sync_and_async_logprobs(self, wants_logprobs):
        """In-graph NKI: sync returns the logprobs, async defers them to get_output."""
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        md = metadata([0.7, 0.0], count=0 if wants_logprobs else None)
        ids = torch.tensor([5, 2, 0], dtype=torch.int32)
        scores = torch.tensor([[-0.25], [-0.75], [0.0]])
        ranks = torch.tensor([[2], [1], [1]], dtype=torch.int32)
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(sampling_metadata=md, num_reqs=3),
            nki_sampling=False,
            nki_in_graph_sampling=True,
            on_device_sampling=True,
            use_async_scheduling=False,
            _on_device_nki_scores=(scores, ranks),
            _pending_nki_logprobs=None,
        )
        out = runner_module.NeuronModelRunner._sample(runner, ids)
        assert out.sampled_token_ids == [[5], [2], [0]]
        if wants_logprobs:
            assert out.logprobs_tensors.logprobs.tolist() == [[-0.25], [-0.75], [0.0]]
            assert out.logprobs_tensors.selected_token_ranks.tolist() == [2, 1, 1]
        else:
            assert out.logprobs_tensors is None

        runner.use_async_scheduling = True
        out = runner_module.NeuronModelRunner._sample(runner, ids)
        assert out.sampled_token_ids is ids and out.logprobs_tensors is None
        if wants_logprobs:
            assert runner._pending_nki_logprobs == (ids, scores, ranks)
        else:
            assert runner._pending_nki_logprobs is None

    def test_async_output_materializes_nki_logprobs_with_the_tokens(self):
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        written = []
        runner = SimpleNamespace(
            _update_batch_state_with_samples=lambda ids,
            snapshot_req_ids: written.append((ids, snapshot_req_ids)),
        )
        ids = torch.tensor([5, 2, 0], dtype=torch.int32)  # one padded row
        scores = torch.tensor([[-0.25], [-0.75], [0.0]])
        ranks = torch.tensor([[2], [1], [1]], dtype=torch.int32)
        output = ModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            sampled_token_ids=ids,
        )
        async_output = runner_module.AsyncNeuronModelRunnerOutput(output, runner)
        async_output.pending_nki_logprobs = (ids, scores, ranks)
        result = async_output.get_output()
        assert result.sampled_token_ids[:2] == [[5], [2]]
        assert result.logprobs.logprobs.tolist() == [[-0.25], [-0.75]]
        assert result.logprobs.logprob_token_ids.tolist() == [[5], [2]]
        assert result.logprobs.sampled_token_ranks.tolist() == [2, 1]
        assert async_output.pending_nki_logprobs is None
        assert result.logprobs.slice_request(1, 1).logprobs.tolist() == [[-0.75]]

    def test_a_discarded_prefill_row_draws_no_uniform(self):
        # A chunked prefill that is not final discards its token. Its seed must
        # not advance, so the tokens do not depend on how the prompt is chunked.
        from vllm_neuron.vllm.worker.neuron_model_runner import (
            build_nki_sampling_params_tensor,
        )

        def batch():
            return SimpleNamespace(
                num_reqs=2,
                temperature_cpu=np.array([0.7, 0.9], dtype=np.float32),
                generators={
                    0: torch.Generator().manual_seed(5),
                    1: torch.Generator().manual_seed(6),
                },
            )

        b, ref = batch(), batch()
        params = build_nki_sampling_params_tensor(b, 2, discard_rows={1})
        assert params[1].tolist() == [0.0, 1.0, 1.0]
        assert params[0, 0] == torch.rand((), generator=ref.generators[0])
        assert torch.equal(b.generators[1].get_state(), ref.generators[1].get_state())

    def test_the_separate_sampler_restores_a_discarded_row_seed(self):
        from vllm_neuron.vllm.worker import neuron_model_runner as runner_module

        md = metadata([0.7, 1.0])
        state = md.generators[1].get_state()

        def sample(logits, sampling_metadata):
            for generator in sampling_metadata.generators.values():
                torch.rand((), generator=generator)
            return "nki-result"

        runner = self._runner(md, Mock(side_effect=sample), Mock(), 32)
        runner._nki_discard_rows = {1}
        group = SimpleNamespace(rank_in_group=0, world_size=1)
        with patch.object(runner_module, "get_tp_group", return_value=group):
            assert (
                runner_module.NeuronModelRunner._sample(runner, torch.zeros(2, 32))
                == "nki-result"
            )
        assert torch.equal(md.generators[1].get_state(), state)
        assert not torch.equal(
            md.generators[0].get_state(), metadata([0.7, 1.0]).generators[0].get_state()
        )

    def test_in_graph_padding_rows_are_greedy(self):
        from vllm_neuron.vllm.worker.neuron_model_runner import (
            build_nki_sampling_params_tensor,
        )

        batch = SimpleNamespace(
            num_reqs=0, temperature_cpu=np.zeros(4, np.float32), generators={}
        )
        state = torch.random.get_rng_state()
        params = build_nki_sampling_params_tensor(batch, 2)
        assert params.tolist() == [[0.0, 1.0, 1.0]] * 2
        assert torch.equal(torch.random.get_rng_state(), state)

    @pytest.mark.skipif(
        os.environ.get("SAMPLER_INTEGRATION_HARDWARE") != "1",
        reason="requires the Neuron torch.compile backend",
    )
    def test_warmup_prevents_serving_compiles(self, monkeypatch):
        """All live batch sizes reuse native graphs, including mixed sampling."""
        from torch._dynamo.backends.registry import lookup_backend
        from vllm_neuron.envs import get_compile_backend_name

        torch._dynamo.reset()
        compile_fn = torch.compile
        backend = lookup_backend(get_compile_backend_name())
        compilations = []

        def counted_backend(graph, inputs, **kwargs):
            compilations.append(graph)
            return backend(graph, inputs, **kwargs)

        def compile_counted(fn, **kwargs):
            kwargs["backend"] = counted_backend
            return compile_fn(fn, **kwargs)

        monkeypatch.setattr(torch, "compile", compile_counted)
        sampler = NeuronNKISampler()
        logits = (
            torch.randn(8, 248320, generator=torch.Generator().manual_seed(71))
            .to(torch.bfloat16)
            .to("neuron:0")
        )
        rng = torch.random.get_rng_state()
        with torch._dynamo.config.patch(recompile_limit=64):
            sampler.warmup(logits[:1], 8)
            assert torch.equal(torch.random.get_rng_state(), rng)
            assert len(compilations) == 8
            for batch in [8, 3, 7, 1, 5, 2, 6, 4]:
                for temps in ([0.0] * batch, [0.0] + [0.7] * (batch - 1)):
                    output = sampler(logits[:batch], metadata(temps))
                    assert output.sampled_token_ids.shape == (batch, 1)
                    values = logits[:batch].cpu().float()
                    assert output.sampled_token_ids[0, 0] == values[0].argmax()
                    temperatures = torch.tensor(temps)
                    temperatures[temperatures < 1e-5] = 1
                    expected = torch.log_softmax(
                        values / temperatures[:, None], dim=-1
                    ).gather(1, output.sampled_token_ids.long())
                    torch.testing.assert_close(
                        output.logprobs_tensors.logprobs,
                        expected,
                        atol=5e-5,
                        rtol=2e-5,
                    )
            assert len(compilations) == 8, "Serving compiled an unwarmed graph"
        # Do not leave eight guarded variants consuming the cache budget of
        # the other adapter tests, which compile different dtypes/score modes.
        torch._dynamo.reset()

    @pytest.mark.skipif(
        os.environ.get("SAMPLER_INTEGRATION_HARDWARE") != "1",
        reason="requires a Neuron device and the Neuron torch.compile backend",
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
    def test_nki_serving_bridge_full_vocabulary(self, dtype, mode):
        """Exercise the production adapter, compiler and device tensor path."""
        import numpy as np

        logits = torch.randn(
            4, 248320, generator=torch.Generator().manual_seed(450)
        ).to(dtype)
        md = metadata([0.0, 0.7, 1.5, 1.0])
        sampler = NeuronNKISampler(logprobs_mode=mode)
        output = sampler(logits.to("neuron:0"), md)
        temps = np.array([[1.0], [0.7], [1.5], [1.0]], np.float32)
        if mode == "raw_logprobs":
            temps[:] = 1
        probs = reference_probabilities(logits.float().numpy(), temps)
        ids = output.sampled_token_ids[:, 0].long()
        assert ids[0].item() == logits[0].argmax().item()
        expected = np.log(probs[np.arange(4), ids.numpy()])
        np.testing.assert_allclose(
            output.logprobs_tensors.logprobs[:, 0].numpy(),
            expected,
            atol=5e-5,
            rtol=2e-5,
        )
        np.testing.assert_array_equal(
            output.logprobs_tensors.selected_token_ranks.numpy(),
            (probs >= probs[np.arange(4), ids.numpy(), None]).sum(-1),
        )
