# SPDX-License-Identifier: Apache-2.0
"""Register the Qwen3.5 architectures with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in this
folder), so vLLM runs it in every process it starts (the frontend, the engine
core and each worker, also under ``vllm serve``) before it resolves the
architecture. An architecture that is registered only in a user's script
exists only in that process and in the processes it forks.
"""

ARCHITECTURES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)


def register() -> None:
    from vllm import ModelRegistry

    for arch in ARCHITECTURES:
        ModelRegistry.register_model(arch, f"qwen3_5.factory:{arch}")
    # Every process vllm serve starts validates the config, so the relaxation has
    # to be applied here too, not only in scripts/run.py.
    relax_head_divisibility()


def relax_head_divisibility() -> None:
    """Let the world be wider than the model's attention-head count.

    ``ModelConfig.verify_with_parallel_config`` rejects
    ``total_num_attention_heads % tensor_parallel_size != 0``, so a 32-head model
    cannot get 64 ranks even though the model does not need one head per rank.
    vLLM already replicates in the analogous case: ``get_num_kv_heads`` is
    ``max(1, total // tp)``. Only the query head count is checked for exact
    divisibility.

    Qwen3.5-397B-A17B needs 64 ranks to fit its weights (738 GiB of text-only
    parameters against 23.5 GiB per logical core) and has 32 query heads. The
    modules replicate the affected heads across ``tp // heads`` consecutive
    ranks (see ``Qwen3_5Attention.__init__`` and
    ``Qwen3_5GatedDeltaNet.__init__``).

    Relaxed only in the safe direction: the world must be a whole multiple of
    the head count, so every rank's shapes equal those of a legal narrower world.
    """
    from vllm.config.model import ModelConfig

    if getattr(ModelConfig, "_qwen3_5_relaxed_heads", False):
        return
    original_verify = ModelConfig.verify_with_parallel_config

    def verify(self, parallel_config):
        arch = self.model_arch_config
        heads, tp = arch.total_num_attention_heads, parallel_config.tensor_parallel_size
        if heads and tp > heads and tp % heads == 0:
            # Present a count the check accepts, for the duration of the check.
            try:
                arch.total_num_attention_heads = tp
                return original_verify(self, parallel_config)
            finally:
                arch.total_num_attention_heads = heads
        return original_verify(self, parallel_config)

    def get_num_attention_heads(self, parallel_config):
        heads = self.model_arch_config.total_num_attention_heads
        return max(1, heads // parallel_config.tensor_parallel_size)

    ModelConfig.verify_with_parallel_config = verify
    ModelConfig.get_num_attention_heads = get_num_attention_heads
    ModelConfig._qwen3_5_relaxed_heads = True
