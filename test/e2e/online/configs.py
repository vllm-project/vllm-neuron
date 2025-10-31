# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from test.e2e.online.online_server_runner import OnlineCfg

# TEST_DIR should point to test/ directory (not test/e2e/)
TEST_DIR = Path(__file__).resolve().parent.parent.parent
# ----------------------------------------------------------------------
# Tiny Test
# ----------------------------------------------------------------------
TINYTEST_CONFIG = OnlineCfg(
    name="online-inference-tinytest",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tp_degree=32,
    batch_size=4,
    max_model_len=128,
    max_tokens=64,
    port=8000,
    accuracy_check=False,  # if we want to verify outputs vs HF goldens
)

# ----------------------------------------------------------------------
# Integration configs
# ----------------------------------------------------------------------
INTEGRATION_CONFIGS = [
    OnlineCfg(
        name="llama-32-1B_integration_bs1",
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_degree=32,
        batch_size=1,
        max_model_len=128,
        max_tokens=64,
        port=8000,
        accuracy_check=True,
    ),
    OnlineCfg(
        name="llama-32-1B_integration_bs4",
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_degree=32,
        batch_size=4,
        max_model_len=128,
        max_tokens=64,
        port=8000,
        accuracy_check=True,
    ),
    OnlineCfg(
        name="open_llama_7b_bs8",
        model="openlm-research/open_llama_7b",
        tp_degree=32,
        batch_size=8,
        max_model_len=128,
        max_tokens=64,
        port=8000,
        accuracy_check=False,
    ),
    OnlineCfg(
        name="Qwen3-8B_bs16",
        model="Qwen/Qwen3-8B",
        tp_degree=32,
        batch_size=16,
        max_model_len=128,
        max_tokens=64,
        port=8000,
        accuracy_check=False,
    ),
]

# ----------------------------------------------------------------------
# Qwen test configuration
# ----------------------------------------------------------------------
QWEN_CONFIG = OnlineCfg(
    name="qwen25-7b-instruct",
    model="Qwen/Qwen2.5-7B-Instruct",
    tp_degree=32,  # This with 28 attention heads should trigger the override
    batch_size=1,
    max_model_len=128,
    max_tokens=50,
    port=8000,
    accuracy_check=
    False,  # Focus on testing that server starts without validation error
)

# ----------------------------------------------------------------------
# Eagle speculative decoding configs
# ----------------------------------------------------------------------
EAGLE_CONFIGS = [
    # Baseline (no APC/CP)
    OnlineCfg(
        name="llama-8B-eagle-bs1",
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_degree=32,
        batch_size=1,
        max_model_len=256,
        max_tokens=64,
        port=8000,
        accuracy_check=False,
        draft_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        num_speculative_tokens=5,
        speculation_type="eagle",
    ),
    OnlineCfg(
        name="llama-8B-eagle-bs4",
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp_degree=32,
        batch_size=4,
        max_model_len=256,
        max_tokens=64,
        port=8000,
        accuracy_check=False,
        draft_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        num_speculative_tokens=5,
        speculation_type="eagle",
    ),
]

# ----------------------------------------------------------------------
# Tool calling configs
# ----------------------------------------------------------------------
TOOL_CALLING_CONFIGS = [
    OnlineCfg(
        name="llama-70B-tool-bs1",
        model="meta-llama/Llama-3.1-70B",
        tp_degree=32,
        batch_size=1,
        max_model_len=512,
        max_tokens=210,
        port=8000,
        accuracy_check=False,
        enable_auto_tool_choice=True,
        tool_call_parser="llama3_json",
        custom_chat_template_path=str(
            TEST_DIR / "utils" / "server" /
            "tool_chat_template_llama3.1_json.jinja"),
        use_chat_url=True
    ),  # tool calling is only available for online serving's chat completion endpoint)
]
