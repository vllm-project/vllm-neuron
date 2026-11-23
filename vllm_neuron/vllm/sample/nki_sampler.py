# SPDX-License-Identifier: Apache-2.0
"""Full-vocabulary NKI sampler for synchronous Neuron inference."""

import torch
from vllm.logger import init_logger
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

logger = init_logger(__name__)


class NeuronNKISampler(torch.nn.Module):
    """Full-vocabulary device sampling with request-owned host RNG.

    This synchronous path supports temperature and mixed greedy
    rows. Requests are validated by the platform before scheduling. Only the
    compact token, selected raw/processed logprob and rank come back to the host.
    """

    def __init__(self, logprobs_mode="processed_logprobs"):
        super().__init__()
        self.logprobs_mode = logprobs_mode
        self._kernel = None
        self._warmed_shapes = set()

    def warmup(self, logits, max_batch_size):
        """Compile every live batch size before serving, without advancing RNG."""
        signature = (max_batch_size, logits.shape[1], logits.dtype, logits.device)
        if signature in self._warmed_shapes:
            return
        if not 1 <= max_batch_size <= 128:
            raise ValueError("NKI sampling requires max_num_seqs <= 128")
        dummy_logits = logits.new_zeros((max_batch_size, logits.shape[1]))
        for batch in range(1, max_batch_size + 1):
            uniforms = torch.zeros(
                (batch, 1), dtype=torch.float32, device=logits.device
            )
            temperatures = torch.ones_like(uniforms)
            greedy = torch.ones((batch, 1), dtype=torch.int32, device=logits.device)
            for output in self._run_kernel(
                dummy_logits[:batch], uniforms, temperatures, greedy
            ):
                output.cpu()
            logger.info(
                "NKI sampler warmup completed: batch=%d, dtype=%s", batch, logits.dtype
            )
        self._warmed_shapes.add(signature)

    def _run_kernel(self, logits, uniforms, temperatures, greedy):
        if self._kernel is None:
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            from vllm_neuron.envs import get_compile_backend_name
            from vllm_neuron.functional.stochastic_sampling import (
                stochastic_sample_kernel,
            )

            wrapped = wrap_nki(stochastic_sample_kernel)
            raw_logprobs = self.logprobs_mode == "raw_logprobs"

            def invoke(logits, uniforms, temperatures, greedy):
                return wrapped(
                    logits,
                    uniforms,
                    temperatures,
                    greedy_rows=greedy,
                    return_ranks=True,
                    raw_logprobs=raw_logprobs,
                )

            self._kernel = torch.compile(
                invoke,
                backend=get_compile_backend_name(),
                fullgraph=True,
                dynamic=False,
                options={
                    "compiler_args": [
                        "--auto-cast=none",
                        (
                            "--internal-backend-options=--enable-verifier=false "
                            "--enable-nested-dynamic-loop"
                        ),
                    ]
                },
            )
        # The runner executes host bookkeeping under force_eager. Only this
        # device graph needs compilation; do not compile the RNG/metadata code.
        with torch.compiler.set_stance("default"):
            return self._kernel(logits, uniforms, temperatures, greedy)

    def forward(self, logits, sampling_metadata):
        md = sampling_metadata
        logits = logits.contiguous()
        batch, vocab = logits.shape
        if not 1 <= batch <= 128 or not 1 <= vocab <= 128 * 4096:
            raise ValueError("NKI sampling requires B <= 128 and V <= 524288")
        if logits.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("NKI sampling requires FP32 or BF16 logits")
        # Platform validation owns request constraints; these assert the
        # dispatch/metadata contract before consuming request RNG.
        assert not has_sampling_filters(md, vocab), (
            "Filtered batches must use CPU sampling"
        )
        assert md.all_greedy or md.temperature is not None
        temperatures = (
            torch.zeros(batch, dtype=torch.float32)
            if md.all_greedy
            else md.temperature.cpu().float().clone()
        )
        greedy = temperatures < 1e-5
        temperatures[greedy] = 1
        # Consume one uniform per stochastic request, and none for greedy rows.
        # TP rank zero alone calls this method; peers receive its final decision.
        uniforms = torch.zeros(batch, dtype=torch.float32)
        for row in (~greedy).nonzero(as_tuple=True)[0].tolist():
            uniforms[row] = torch.rand((), generator=md.generators.get(row))
        device = logits.device
        args = [
            value.reshape(batch, 1).to(device)
            for value in (uniforms, temperatures, greedy.int())
        ]
        ids, scores, ranks = (x.cpu() for x in self._run_kernel(logits, *args))
        if (ids < 0).any() or (ids >= vocab).any() or not torch.isfinite(scores).all():
            raise ValueError("NKI sampling encountered invalid logits or empty support")
        logprobs = (
            LogprobsTensors(ids.int(), scores.float(), ranks.reshape(-1).long())
            if md.max_num_logprobs is not None or bool(md.logprob_token_ids)
            else None
        )
        return SamplerOutput(ids.int(), logprobs)


def has_sampling_filters(metadata, vocab_size):
    """Whether any row needs filtering; vLLM uses vocab_size for disabled k."""
    if metadata.top_k is not None:
        k = metadata.top_k.cpu()
        if ((k > 0) & (k < vocab_size)).any():
            return True
    return metadata.top_p is not None and bool((metadata.top_p.cpu() < 1).any())


def validate_nki_sampling_params(params, logprobs_mode="processed_logprobs"):
    """Validate the NKI route, whose top-k/top-p requests use vLLM on CPU."""
    unsupported = [
        name
        for name in (
            "min_p",
            "min_tokens",
            "logit_bias",
            "structured_outputs",
        )
        if getattr(params, name, None)
    ]
    if params.prompt_logprobs is not None:
        unsupported.append("prompt_logprobs")
    if getattr(params, "thinking_token_budget", None) is not None:
        unsupported.append("thinking_token_budget")
    unsupported.extend(
        name
        for name, default in (
            ("frequency_penalty", 0),
            ("presence_penalty", 0),
            ("repetition_penalty", 1),
            ("allowed_token_ids", None),
            ("bad_words", None),
            ("logprob_token_ids", None),
        )
        if getattr(params, name, default) not in (default, None, [])
    )
    if params.logprobs not in (None, 0):
        unsupported.append("logprobs other than 0")
    if params.logprobs is not None and logprobs_mode not in (
        "raw_logprobs",
        "processed_logprobs",
    ):
        unsupported.append("score modes other than raw_logprobs or processed_logprobs")
    if unsupported:
        raise ValueError(
            "Neuron NKI sampling does not yet support: " + ", ".join(unsupported)
        )
