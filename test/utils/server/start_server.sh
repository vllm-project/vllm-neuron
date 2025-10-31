#!/usr/bin/env bash
set -euo pipefail

model_dir=${1:?}
port=${2:-8000}
cores=${3:-0-31}
max_model_len=${4:-2048}
batch_size=${5:-32}
tp_size=${6:-32}
n_threads=${7:-32}
shift 7 || true
set -x

override_neuron_config="{}"
enable_chunked_prefill=false
enable_prefix_caching=false
speculative_config=""
chat_template=""
enable_lora=false
lora_modules=""
max_loras=4
max_cpu_loras=0
max_lora_rank=256

# Treat these as "unset" unless provided; only forward if > 0
max_num_batched_tokens=0
block_size=0
num_gpu_blocks_override=0
enable_auto_tool_choice=false
tool_call_parser=""
reasoning_parser=""

while [[ "${#}" -gt 0 ]]; do
  case "$1" in
    --speculative-config) speculative_config="$2"; shift 2 ;;
    --chat-template)      chat_template="$2";      shift 2 ;;
    --enable-chunked-prefill) enable_chunked_prefill=true; shift ;;
    --no-enable-chunked-prefill) enable_chunked_prefill=false; shift ;;
    --max-num-batched-tokens) max_num_batched_tokens="$2"; shift 2 ;;
    --block-size) block_size="$2"; shift 2 ;;
    --num-gpu-blocks-override) num_gpu_blocks_override="$2"; shift 2 ;;
    --enable-prefix-caching) enable_prefix_caching=true; shift ;;
    --no-enable-prefix-caching) enable_prefix_caching=false; shift ;;
    --override-neuron-config) override_neuron_config="$2"; shift 2 ;;
    --enable-lora) enable_lora=true; shift ;;
    --lora-modules)
      shift
      mods=()
      while [[ "${#}" -gt 0 && "$1" != --* ]]; do mods+=("$1"); shift; done
      lora_modules="${mods[*]}"
      ;;
    --max-lora-rank) max_lora_rank="$2"; shift 2 ;;
    --max-loras)     max_loras="$2";     shift 2 ;;
    --max-cpu-loras) max_cpu_loras="$2"; shift 2 ;;
    --enable-auto-tool-choice) enable_auto_tool_choice=true; shift ;;
    --tool-call-parser) tool_call_parser="$2"; shift 2 ;;
    --reasoning-parser) reasoning_parser="$2"; shift 2 ;;
    *) echo "Unknown param: $1"; exit 1 ;;
  esac
done

# helper: case-insensitive truthy
is_true() {
  case "${1:-}" in
    [Tt][Rr][Uu][Ee]|1|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

cmd_args=(
  --model "$model_dir"
  --tensor-parallel-size "$tp_size"
  --max-model-len "$max_model_len"
  --max-num-seqs "$batch_size"
  --port "$port"
)

[[ -n "$speculative_config" ]] && cmd_args+=(--speculative-config "$speculative_config")

# -------- Chunked Prefill (CP) --------
if is_true "$enable_chunked_prefill"; then
  cmd_args+=(--enable-chunked-prefill)
  # Only pass knobs if explicitly set to > 0
  if [[ "${max_num_batched_tokens:-0}" -gt 0 ]]; then
    cmd_args+=(--max-num-batched-tokens "$max_num_batched_tokens")
  fi
  if [[ "${block_size:-0}" -gt 0 ]]; then
    cmd_args+=(--block-size "$block_size")
  fi
  if [[ "${num_gpu_blocks_override:-0}" -gt 0 ]]; then
    cmd_args+=(--num-gpu-blocks-override "$num_gpu_blocks_override")
  fi
else
  cmd_args+=(--no-enable-chunked-prefill)
fi

# -------- Prefix Caching (APC) --------
if is_true "$enable_prefix_caching"; then
  cmd_args+=(--enable-prefix-caching)
  # Only pass knobs if explicitly set to > 0
  if [[ "${block_size:-0}" -gt 0 ]]; then
    cmd_args+=(--block-size "$block_size")
  fi
  if [[ "${num_gpu_blocks_override:-0}" -gt 0 ]]; then
    cmd_args+=(--num-gpu-blocks-override "$num_gpu_blocks_override")
  fi
else
  cmd_args+=(--no-enable-prefix-caching)
fi

# -------- LoRA --------
if is_true "$enable_lora"; then
  read -r -a larr <<< "$lora_modules"
  cmd_args+=(--enable-lora --max-loras "$max_loras" --max-cpu-loras "$max_cpu_loras" --max-lora-rank "$max_lora_rank" --lora-modules "${larr[@]}")
fi

if [[ -n "${override_neuron_config:-}" && "${override_neuron_config}" != "{}" ]]; then
  # wrap override_neuron_config as a sub-dict under additional_config
  additional_config=$(printf '{"override_neuron_config":%s}' "$override_neuron_config")
  cmd_args+=(--additional-config "$additional_config")
fi
[[ -n "$chat_template" ]] && cmd_args+=(--chat-template "$chat_template")
if is_true "$enable_auto_tool_choice"; then
  cmd_args+=(--enable-auto-tool-choice)
fi
[[ -n "$tool_call_parser" ]] && cmd_args+=(--tool-call-parser "$tool_call_parser")

# -------- Reasoning Outputs --------
[[ -n "$reasoning_parser" ]] && cmd_args+=(--reasoning-parser "$reasoning_parser")

export NEURON_RT_DBG_RDH_CC=0
export NEURON_RT_INSPECT_ENABLE=0
export XLA_HANDLE_SPECIAL_SCALAR=1
export UNSAFE_FP8FNCAST=1
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"

python3 -m vllm.entrypoints.openai.api_server "${cmd_args[@]}"