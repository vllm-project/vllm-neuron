# SPDX-License-Identifier: Apache-2.0
"""
Client-side runner for sending prompts to the online inference server
and validating responses. Handles retries, payload construction, and logging.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from test.utils.server.server import VllmServer
from typing import List, Optional

import requests

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import transformers for the optional accuracy step
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_OK = True
except Exception as _exc:
    _HF_OK = False
    _HF_IMPORT_ERR = _exc


# --- add these fields to OnlineCfg ---
@dataclass
class OnlineCfg:
    name: str
    model: str
    tp_degree: int = 32
    batch_size: int = 4
    max_model_len: int = 128
    max_tokens: int = 16
    port: int = 8000
    accuracy_check: Optional[bool] = None
    override_neuron_config: dict = None
    use_chat_url: bool = False

    # Speculative Decoding
    draft_model_path: Optional[str] = None
    num_speculative_tokens: int = 5
    speculation_type: Optional[str] = "eagle"

    # Prefix Caching (APC)
    enable_prefix_caching: bool = False
    block_size: int = 32  # optional; 0 lets server default
    num_gpu_blocks_override: int = 16

    # Chunked Prefill (CP)
    enable_chunked_prefill: bool = False
    max_num_batched_tokens: int = 0
    prefill_block_size: int = 0

    # Tool Calling
    enable_auto_tool_choice: bool = False
    tool_call_parser: str = None
    custom_chat_template_path: str = None


def _post_with_retries(
    url: str,
    payload: dict,
    headers: dict,
    tries: int = 10,
    sleep_s: int = 3,
):
    last_exc = None
    for _ in range(tries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.ok:
                return r
            else:
                logger.warning("POST %s -> %s: %s", url, r.status_code, r.text)
        except requests.RequestException as e:
            last_exc = e
            logger.warning("POST %s failed with %r; retrying in %ss", url, e,
                           sleep_s)
        time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise RuntimeError("POST failed after retries")


# -------------------------- HF CHECK -------------------------- #
def _run_hf_fulltext_prefix_check(
        served_model_fs_path: str,
        prompts: List[str],
        max_model_len: int,
        actual_full_texts: List[
            str],  # FULL strings: prompt + server continuation
) -> None:

    if not _HF_OK:
        logger.warning(
            "Skipping HF accuracy check (transformers import failed: %r)",
            _HF_IMPORT_ERR)
        return

    logger.info("Running HF full-text prefix check with model at: %s",
                served_model_fs_path)

    tok = AutoTokenizer.from_pretrained(served_model_fs_path,
                                        padding_side="left")
    if getattr(tok, "pad_token_id", None) is None and getattr(
            tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(served_model_fs_path)
    model.eval()

    batch = tok(prompts, return_tensors="pt", padding=True)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            do_sample=False,
            top_k=1,
            max_length=max_model_len,
        )

    expected_full_texts = tok.batch_decode(gen_ids, skip_special_tokens=True)

    # Compare (expected is HF, actual is server)
    for i, (actual,
            expected) in enumerate(zip(actual_full_texts,
                                       expected_full_texts)):
        if not expected.startswith(actual):
            logger.error(
                "HF prefix mismatch at idx=%d\n"
                "  PROMPT:    %r\n"
                "  ACTUAL:    %r\n"
                "  EXPECTED:  %r", i, prompts[i], actual[:300], expected[:300])
            raise AssertionError(f"HF prefix mismatch at idx={i}:\n"
                                 f"  actual:   {actual!r}\n"
                                 f"  expected: {expected!r}")


# ------------------------------- MAIN RUNNER -------------------------------- #
def run_online_integration(cfg: OnlineCfg) -> None:
    # Log the config
    logger.info("OnlineCfg: %s", json.dumps(asdict(cfg), indent=2))

    # Handle EAGLE draft model conversion if needed
    model_path = cfg.model
    draft_model_path = cfg.draft_model_path

    if draft_model_path and cfg.speculation_type == "eagle":
        from test.utils.eagle_nxdi_util import fix_eagle_draft_for_nxdi
        from test.utils.fsx_utils.model_path import resolve_model_dir

        target_model_dir, _ = resolve_model_dir(cfg.model)
        draft_model_dir, _ = resolve_model_dir(draft_model_path)

        if not os.path.isabs(draft_model_dir):
            logger.info(
                "Downloading EAGLE models from HuggingFace and converting")
            from huggingface_hub import snapshot_download

            # Download both models
            target_model_dir = snapshot_download(cfg.model)
            draft_model_dir = snapshot_download(draft_model_path)

            logger.info("Downloaded target model to: %s", target_model_dir)
            logger.info("Downloaded draft model to: %s", draft_model_dir)

            # Convert EAGLE draft model for Neuron compatibility
            logger.info(
                "Converting EAGLE draft model for Neuron compatibility...")
            fix_eagle_draft_for_nxdi(target_model_dir, draft_model_dir)
            logger.info("EAGLE draft model conversion complete")

            # Update paths to use the converted local paths
            model_path = target_model_dir
            draft_model_path = draft_model_dir
        else:
            logger.info("Using cached EAGLE models (already converted)")

    # Spin up the server
    server = VllmServer(
        name=cfg.name,
        model_path=model_path,
        batch_size=cfg.batch_size,
        tp_degree=cfg.tp_degree,
        n_vllm_threads=32,
        server_port=cfg.port,
        override_neuron_config=cfg.override_neuron_config,
        max_model_len=cfg.max_model_len,

        # CP/APC
        enable_chunked_prefill=cfg.enable_chunked_prefill,
        enable_prefix_caching=cfg.enable_prefix_caching,
        chunk_size=cfg.max_num_batched_tokens,
        block_size=(cfg.prefill_block_size or cfg.block_size),
        num_blocks_override=cfg.num_gpu_blocks_override,

        # Speculation
        draft_model_path=draft_model_path,
        speculation_len=cfg.num_speculative_tokens,
        speculation_type=cfg.speculation_type if draft_model_path else None,

        # Tool Calling
        enable_auto_tool_choice=cfg.enable_auto_tool_choice,
        tool_call_parser=cfg.tool_call_parser,
        custom_chat_template_path=cfg.custom_chat_template_path)
    port, proc, healthy = server.start()
    try:
        assert healthy, "vLLM server did not become healthy"
        if cfg.use_chat_url:
            url = f"http://localhost:{port}/v1/chat/completions"
        else:
            url = f"http://localhost:{port}/v1/completions"
        headers = {"Content-Type": "application/json;charset=UTF-8"}

        # ---- fixed prompts (repeat to match batch size) ----
        if cfg.use_chat_url:
            base_prompts: List[str] = [
                {
                    "role":
                    "user",
                    "content":
                    "I am going to count from 1 to 250 only once. I will not say anything further. 1,2,3,4,"
                },
                {
                    "role": "user",
                    "content": "The president of the United States is"
                },
            ]
        else:
            base_prompts: List[str] = [
                "I am going to count from 1 to 250 only once. I will not say anything further. 1,2,3,4,",
                "The president of the United States is",
            ]
        if cfg.batch_size <= len(base_prompts):
            prompts = base_prompts[:cfg.batch_size]
        else:
            reps = (cfg.batch_size + len(base_prompts) -
                    1) // len(base_prompts)
            prompts = (base_prompts * reps)[:cfg.batch_size]

        if cfg.tool_call_parser:

            def add(a: int, b: int):
                return f"Getting the addition for {a} and {b}..."

            tool_functions = {"add": add}
            messages = [{"role": "user", "content": "What's 2+3?"}]
            tools = [{
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number"
                            },
                            "b": {
                                "type": "number"
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            }]
            prompts = [json.dumps((messages + tools)[0])]
            print(f"{prompts=}, {type(prompts)=}")

        served_model = getattr(server, "served_model_id", cfg.model)

        # ---- context-safe clamping of max_tokens ----
        try:
            if _HF_OK:
                tok = AutoTokenizer.from_pretrained(served_model,
                                                    padding_side="left")
                if getattr(tok, "pad_token_id", None) is None and getattr(
                        tok, "eos_token_id", None) is not None:
                    tok.pad_token_id = tok.eos_token_id
                if cfg.tool_call_parser:
                    ids = tok.apply_chat_template(
                        messages,
                        tool=tools,
                        chat_template=cfg.custom_chat_template_path,
                        add_special_tokens=False)
                    max_prompt_len = len(ids)
                else:
                    ids = tok(prompts, add_special_tokens=False)["input_ids"]
                    max_prompt_len = max(len(x) for x in ids)
            else:
                tok = None
                max_prompt_len = max(1, max(len(p) for p in prompts) // 4)

            # Leave a tiny margin; align with your earlier logic
            allowed_new = cfg.max_model_len - max_prompt_len - 2
            if allowed_new <= 0:
                raise AssertionError(
                    f"[{cfg.name}] Prompt contains {max_prompt_len} tokens; "
                    f"max_model_len={cfg.max_model_len}, margin=2. No room for generation."
                )
            safe_max_new = max(1, min(cfg.max_tokens, allowed_new))
            if safe_max_new < cfg.max_tokens:
                logger.info(
                    "[%s] Clamping max_tokens from %d -> %d (max_model_len=%d, prompt=%d, margin=%d)",
                    cfg.name, cfg.max_tokens, safe_max_new, cfg.max_model_len,
                    max_prompt_len, 2)
        except Exception as e:
            logger.warning(
                "Token length estimation failed (%r); using cfg.max_tokens=%d",
                e, cfg.max_tokens)
            safe_max_new = cfg.max_tokens
            max_prompt_len = None  # unknown

        payload = {
            "model": served_model,
            "max_tokens": safe_max_new,  # server new-token cap
            "temperature": 0,  # greedy
            "top_p": 1,
            "top_k": 1,
        }
        if cfg.use_chat_url:
            payload.update({"messages": prompts})
        else:
            payload.update({"prompt": prompts})
        if cfg.tool_call_parser:
            payload.update({
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            })

        logger.info("Sending %d prompts to %s", len(prompts), url)
        for i, p in enumerate(prompts):
            logger.debug("Prompt[%d]: %s", i, p)

        resp = _post_with_retries(url, payload, headers)
        data = resp.json()

        # Validate shape
        assert "choices" in data, f"bad response: {json.dumps(data)}"
        assert len(data["choices"]) == cfg.batch_size, "batch size mismatch"

        # Build FULL outputs to compare (prompt + continuation)
        actual_full_texts: List[str] = []
        for i, choice in enumerate(data["choices"]):
            cont = choice.get("text", "")
            sanitized = cont[:120].replace("\n", "\\n")
            print(f"Output[{i}]: {sanitized}")
            if cfg.tool_call_parser:
                tool_call = choice["message"]["tool_calls"]
                if len(tool_call) > 0:
                    tool_call = tool_call[0]["function"]
                    print(
                        f"Tool call result[{i}]: {tool_functions[tool_call['name']](**json.loads(tool_call['arguments']))}"
                    )
            actual_full_texts.append(str(prompts[i]) + cont)

        # Type sanity check
        for c in data["choices"]:
            assert isinstance(c.get("text", ""),
                              str), "choice.text must be a string"

        # Optional accuracy check (HF) with per-test override
        env_default = os.environ.get("ACCURACY_CHECK",
                                     "1").strip().lower() not in ("0", "false",
                                                                  "no", "off",
                                                                  "")
        do_acc = env_default if cfg.accuracy_check is None else bool(
            cfg.accuracy_check)

        if do_acc:
            _run_hf_fulltext_prefix_check(
                served_model_fs_path=served_model,
                prompts=prompts,
                max_model_len=cfg.max_model_len,
                actual_full_texts=actual_full_texts,
            )
        else:
            logger.info(
                "Skipping HF accuracy check (disabled by %s).",
                "config"
                if cfg.accuracy_check is not None else "env ACCURACY_CHECK",
            )

    finally:
        try:
            server.kill_process_and_children(proc.pid)
        except Exception as e:
            logger.warning("Failed to kill server process: %r", e)
