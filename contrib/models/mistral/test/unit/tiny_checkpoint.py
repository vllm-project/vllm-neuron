# SPDX-License-Identifier: Apache-2.0
"""Build a tiny random-weight checkpoint with the Mistral-7B-v0.3 layout.

Used to run the module and integration tests in CPU mode
(``VLLM_NEURON_CPU_MODE=1``) in seconds, without the 14.5 GB checkpoint.

Kept from Mistral-7B-v0.3:
  * architecture ``MistralForCausalLM``, vocab 32768, the real tokenizer
  * default RoPE with rope_theta 1e6, no sliding window, untied embeddings, BF16
  * head_dim == hidden_size // num_attention_heads (128, as in 7B)
  * no ``head_dim`` key in config.json, exactly like the real file

Shrunk: 2 layers, hidden 512, 4 query heads, 2 KV heads, MLP 1024. That is
about 36M parameters (72 MB in BF16), mostly embedding and LM head. Use
tensor parallel size 1 or 2 (the 2 KV heads must split evenly).

    python contrib/models/mistral/test/unit/tiny_checkpoint.py /tmp/tiny-mistral
"""

import argparse
import json
from pathlib import Path

TOKENIZER_SOURCE = "mistralai/Mistral-7B-Instruct-v0.3"


def _rope_theta(config) -> float:
    """Read rope_theta from either transformers v4 or v5 config layout."""
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters:
        return rope_parameters["rope_theta"]
    return config.rope_theta


def build(out_dir: str, tokenizer_source: str | None = TOKENIZER_SOURCE) -> Path:
    """Save the tiny checkpoint to ``out_dir`` and return its path.

    Args:
        out_dir: Directory to write. Created if missing.
        tokenizer_source: Hugging Face repo to copy the tokenizer from (about
            2 MB, not gated). None skips the tokenizer, which is enough for
            tests that pass prompt token ids directly.
    """
    import torch
    from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM

    config = MistralConfig(
        vocab_size=32768,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        sliding_window=None,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        # Larger than the 0.02 default so the random model's logits are not
        # nearly flat; flat logits make greedy argmax comparisons flaky.
        initializer_range=0.1,
        torch_dtype=torch.bfloat16,
    )
    config.architectures = ["MistralForCausalLM"]
    assert _rope_theta(config) == 1000000.0, "rope_theta was not applied"
    assert config.sliding_window is None

    torch.manual_seed(0)
    model = MistralForCausalLM(config).to(torch.bfloat16).eval()

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if tokenizer_source is not None:
        AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(path)

    # Match the real config.json, which has no head_dim key. This exercises
    # the path where head_dim must be derived from hidden_size.
    config_path = path / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict.pop("head_dim", None)
    config_path.write_text(json.dumps(config_dict, indent=2) + "\n")

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip downloading the tokenizer (offline use).",
    )
    args = parser.parse_args()
    path = build(args.out_dir, None if args.no_tokenizer else TOKENIZER_SOURCE)
    print(f"Saved tiny Mistral checkpoint to {path}")
    print("Files:", sorted(p.name for p in path.iterdir()))


if __name__ == "__main__":
    main()
