# SPDX-License-Identifier: Apache-2.0
"""
Launches a vLLM API server for online inference integration tests.
Typically invoked via start_server.sh with model path and configuration
arguments.
"""

import errno
import json
import os
import signal
import socket
import subprocess
import time
import warnings
from test.utils.fsx_utils.model_path import resolve_model_dir
from typing import Optional, Tuple

import psutil
import requests

LARGE_LLM_LIST = ['dbrx', 'llama-3.3-70B', 'llama-3.1-405b', 'qwen-3-235b']
REQUIRED_SERVER_KEYS = (
    "name",
    "model_path",
    "tp_degree",
    "batch_size",
    "max_model_len",
)


def _get_modules_to_not_quantize(
    local_model_dir: str,
    override_json_path: Optional[str] = None
) -> Tuple[Optional[list], Optional[list]]:
    """
    Loads modules-to-not-convert lists for quantized checkpoints .

    Priority:
      1) override_json_path if provided
      2) <local_model_dir>/modules_to_not_convert.json if present

    Returns:
        (modules_to_not_convert, draft_model_modules_to_not_convert)
        Both values may be None if the file or keys are not found.
    """
    candidate = override_json_path or os.path.join(
        local_model_dir, "modules_to_not_convert.json")
    if not (candidate and os.path.exists(candidate)):
        return None, None

    with open(candidate, "r") as f:
        data = json.load(f)

    modules_to_not_convert = None
    draft_model_modules_to_not_convert = None

    if "model" in data and isinstance(data["model"], dict):
        modules_to_not_convert = data["model"].get("modules_to_not_convert")
    if modules_to_not_convert is None:
        modules_to_not_convert = data.get("modules_to_not_convert")

    if "draft_model" in data and isinstance(data["draft_model"], dict):
        draft_model_modules_to_not_convert = data["draft_model"].get(
            "modules_to_not_convert")

    return modules_to_not_convert, draft_model_modules_to_not_convert


class VllmServer:
    """Managing vLLM server deployment"""

    def __init__(
            self,
            name: str,  # Unique name for the server instance
            model_path: str,  # Logical path: e.g. 'llama-3.1/llama-3.1-8b'
            batch_size: int,
            max_model_len: int = None,
            tp_degree: int = 24,
            n_vllm_threads: int = 32,
            # Quantization (pre-quantized checkpoints)
            is_quantized_checkpoint: bool = False,
            quantization_dtype: Optional[str] = None,
            quantization_type: Optional[str] = None,
            modules_to_not_convert_file: Optional[str] = None,
            # Features
            enable_chunked_prefill: bool = False,
            enable_prefix_caching: bool = False,
            chunk_size: int = 0,
            block_size: int = 0,
            num_blocks_override: int = 0,
            custom_chat_template_path: str = None,
            server_port: int = 8000,
            draft_model_path: str = None,
            speculation_len: int = None,
            speculation_type: str = None,
            compiled_model_path: str = None,
            inference_demo_script: str = None,
            inference_demo_args: str = None,
            override_neuron_config: dict = None,
            **kwargs):
        self.inference_demo_script = None
        if inference_demo_script:
            if not compiled_model_path:
                raise ValueError(
                    "compiled_model_path is required when inference_demo_script is used"
                )
            self.inference_demo_script = inference_demo_script
        self.inference_demo_args = inference_demo_args
        self.name = name
        self.logical_model_path = model_path
        self.max_num_seqs = batch_size
        self.tp_degree = tp_degree
        self.n_vllm_threads = n_vllm_threads
        self.max_model_len = max_model_len

        # Quantization (pre-quantized weights to be provided by user)
        self.is_quantized_checkpoint = is_quantized_checkpoint
        self.quantization_dtype = quantization_dtype
        self.quantization_type = quantization_type
        self.modules_to_not_convert_file = modules_to_not_convert_file

        # Features
        self.enable_chunked_prefill = enable_chunked_prefill
        self.enable_prefix_caching = enable_prefix_caching
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.num_blocks_override = num_blocks_override
        self.compiled_model_path = compiled_model_path
        self.custom_chat_template_path = custom_chat_template_path
        self.server_port = server_port
        self.draft_model_path = draft_model_path
        self.speculation_len = speculation_len
        self.speculation_type = speculation_type
        self.process = None

        self.cores = f"0-{self.tp_degree - 1}"
        self.vllm_tokenizer = kwargs.get('vllm_tokenizer', None)

        self.override_neuron_config = {}
        if override_neuron_config is not None:
            self.override_neuron_config = override_neuron_config

        self.enable_auto_tool_choice = kwargs.get('enable_auto_tool_choice',
                                                  False)
        self.tool_call_parser = kwargs.get('tool_call_parser', None)

    # ---------------- utility ----------------

    def _is_port_available(self, port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("localhost", port))
            return True
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                return False
            # any other bind error: treat as unavailable to be safe
            return False
        finally:
            sock.close()

    def _find_free_port(self, start_port=8000, max_port=65535) -> int:
        for p in range(start_port, max_port):
            if self._is_port_available(p):
                return p
        raise RuntimeError("Unable to find a free port")

    def _health_probe(self,
                      url: str,
                      retries: int,
                      delay: int,
                      process=None) -> bool:
        for _ in range(retries):
            # If the child died at any point, fail fast
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"vLLM engine exited with code {process.returncode}")

            try:
                r = requests.get(url)
                if r.status_code == 200:
                    print("Server is up! Health check passed.")
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(delay)
        print("Server did not respond within the retry limit.")
        return False

    def _retries_for_model(self) -> int:
        # use logical path to infer size
        key = (self.name + self.logical_model_path).lower()
        return 480 if any(s in key for s in (s.lower()
                                             for s in LARGE_LLM_LIST)) else 120

    # ---------------- lifecycle ----------------

    def run_inference_demo(self, resolved_model_dir: str) -> None:
        """Run user-provided compile script with *resolved* local model path."""
        cmd = [
            "/bin/bash",
            self.inference_demo_script,
            resolved_model_dir,  # MODEL_PATH (local, resolved)
            self.compiled_model_path,  # COMPILED_MODEL_PATH
            str(self.tp_degree),  # TP_DEGREE
            str(self.max_model_len),  # SEQ_LEN
            str(self.max_num_seqs),  # BATCH_SIZE
        ]
        # Optional: let explicit args override, if you still want that behavior
        if self.inference_demo_args:
            # NB: this is now *ignored by default*, only used if you really set it
            print(
                "[VllmServer] WARNING: 'inference_demo_args' provided; overriding auto-built arguments."
            )
            cmd = ["/bin/bash", self.inference_demo_script
                   ] + self.inference_demo_args.split()

        # Making sure artifacts go where vLLM will read them
        if self.compiled_model_path:
            os.environ["NEURON_COMPILED_ARTIFACTS"] = self.compiled_model_path

        try:
            result = subprocess.run(cmd, check=True)
            if self.compiled_model_path:
                assert os.path.isdir(self.compiled_model_path) and os.listdir(self.compiled_model_path), \
                    f"No compile artifacts found in {self.compiled_model_path}"
            print(
                f"Inference demo completed successfully. rc={result.returncode}"
            )
        except subprocess.CalledProcessError as e:
            print(f"Inference demo failed with rc={e.returncode}")
            raise

    def start_vllm_server(self):
        health_ok = False
        SERVER_LOG = "vllm_server_out.txt"

        resolved_model_dir, _ = resolve_model_dir(self.logical_model_path)
        self.served_model_id = resolved_model_dir
        print(f"[vLLM] Using model: {resolved_model_dir}")
        print(f"[vLLM] Registered model id: {self.served_model_id}")

        try:
            # optional caches/env
            if self.compiled_model_path:
                os.environ[
                    "NEURON_COMPILED_ARTIFACTS"] = self.compiled_model_path
            if self.vllm_tokenizer:
                os.environ["VLLM_TOKENIZER"] = self.vllm_tokenizer

            if hasattr(self,
                       "inference_demo_script") and self.inference_demo_script:
                self.run_inference_demo(resolved_model_dir)

            # get a free port
            port = self.server_port
            if not self._is_port_available(port):
                print(f"Port {port} is in use. Finding an alternative…")
                port = self._find_free_port(start_port=port)
                print(f"Using port {port}")

            # local start script inside this folder
            start_script = os.path.join(os.path.dirname(__file__),
                                        "start_server.sh")
            if not os.path.exists(start_script):
                raise FileNotFoundError(
                    f"start_server.sh not found at {start_script}")

            args = [
                "/bin/bash",
                start_script,
                resolved_model_dir,
                str(port),
                self.cores,
                str(self.max_model_len),
                str(self.max_num_seqs),
                str(self.tp_degree),
                str(self.n_vllm_threads),
            ]

            # -------- speculation: add CLI JSON, and set fused flags via override -----------
            if self.draft_model_path is not None:
                resolved_draft_dir, _ = resolve_model_dir(
                    self.draft_model_path)
                resolved_draft_dir = resolved_draft_dir.rstrip("/")

                spec_cfg = {
                    "model":
                    resolved_draft_dir,
                    "num_speculative_tokens":
                    self.speculation_len
                    if self.speculation_len is not None else 4,
                    "max_model_len":
                    self.max_model_len,
                    "method": (self.speculation_type or "eagle"),
                }
                # optional but handy for debugging
                print("[vLLM] Speculative config JSON:", json.dumps(spec_cfg))

                args.extend(["--speculative-config", json.dumps(spec_cfg)])

            if self.custom_chat_template_path:
                args.extend(
                    ["--chat-template", self.custom_chat_template_path])

            # -------- Quantized checkpoints ----------
            if self.is_quantized_checkpoint:
                warnings.warn(
                    "is_quantized_checkpoint=True: assuming the provided model "
                    "directory already contains quantized weights and a"
                    "modules_to_not_convert json.",
                    category=UserWarning)
                if self.override_neuron_config is None:
                    self.override_neuron_config = {}

                if self.quantization_dtype:
                    self.override_neuron_config[
                        "quantization_dtype"] = self.quantization_dtype
                if self.quantization_type:
                    self.override_neuron_config[
                        "quantization_type"] = self.quantization_type

                # Load the "do not convert" modules and inject as lists
                mtc, draft_mtc = _get_modules_to_not_quantize(
                    local_model_dir=resolved_model_dir,
                    override_json_path=self.modules_to_not_convert_file,
                )
                if mtc:
                    self.override_neuron_config["modules_to_not_convert"] = mtc
                if draft_mtc:
                    self.override_neuron_config[
                        "draft_model_modules_to_not_convert"] = draft_mtc

            # ------------- feature flags -------------

            # -------- Chunked Prefill (CP) --------
            if self.enable_chunked_prefill:
                args += ["--enable-chunked-prefill"]
                if (self.chunk_size or 0) > 0:
                    args += ["--max-num-batched-tokens", str(self.chunk_size)]
                if (self.block_size or 0) > 0:
                    args += ["--block-size", str(self.block_size)]
                if (self.num_blocks_override or 0) > 0:
                    args += [
                        "--num-gpu-blocks-override",
                        str(self.num_blocks_override)
                    ]
            else:
                args += ["--no-enable-chunked-prefill"]

            # -------- Prefix Caching (APC) --------
            if self.enable_prefix_caching:
                args += ["--enable-prefix-caching"]
                if (self.block_size or 0) > 0:
                    args += ["--block-size", str(self.block_size)]
                if (self.num_blocks_override or 0) > 0:
                    args += [
                        "--num-gpu-blocks-override",
                        str(self.num_blocks_override)
                    ]
            else:
                args += ["--no-enable-prefix-caching"]

            if self.override_neuron_config:
                args += [
                    "--override-neuron-config",
                    json.dumps(self.override_neuron_config)
                ]

            # ------- Tool Calling -----------
            if self.enable_auto_tool_choice:
                args += ["--enable-auto-tool-choice"]
            if self.tool_call_parser and len(self.tool_call_parser) > 0:
                args += ["--tool-call-parser", str(self.tool_call_parser)]

            # launch and log
            with open(SERVER_LOG, "w") as outfile:
                process = subprocess.Popen(args,
                                           text=True,
                                           stdout=outfile,
                                           stderr=outfile)

            # give server some time to start
            time.sleep(60)
            if process.poll() is not None:
                raise RuntimeError(
                    f"subprocess return code: {process.returncode}")

            health_ok = self._health_probe(f"http://localhost:{port}/health",
                                           self._retries_for_model(),
                                           30,
                                           process=process)
        except Exception as e:
            print(f"Failed to start the subprocess: {e}")
            raise RuntimeError(f"Server instantiation failed due to {e}")
        finally:
            try:
                with open(SERVER_LOG, "r") as f:
                    print("\nVllmServer Initialization Logs:")
                    print(f.read())
            except Exception:
                pass

        return port, process, health_ok

    def start(self):
        """
        Starts the vLLM server and performs health checks.

        Returns:
            (int, process, bool): port, process object, health status
        """
        health_url = f"http://localhost:{self.server_port}/health"

        # If something is already listening, try to kill it.
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("Existing server detected, attempting to terminate…")
                self.kill_children_of_process_on_port(self.server_port)
        except requests.RequestException:
            pass

        port, process, ok = self.start_vllm_server()
        if not ok:
            print("Server did not start successfully.")
            raise ConnectionRefusedError("Server did not start successfully.")
        return port, process, ok

    # ------------ teardown helpers ------------

    def kill_process_and_children(self, pid: int):
        try:
            print(f"Terminating process with pid {pid}")
            parent = psutil.Process(pid)
            procs = [parent] + parent.children(recursive=True)
            for p in procs:
                print(f"Sending SIGTERM to PID: {p.pid}")
                p.send_signal(signal.SIGTERM)
            _, alive = psutil.wait_procs(procs, timeout=30)
            for p in alive:
                print(f"PID {p.pid} did not terminate, sending SIGKILL")
                p.send_signal(signal.SIGKILL)
        except Exception as e:
            print(f"Failed to terminate pid={pid}: {e}")

    def kill_children_of_process_on_port(self, port: int):
        pids = {
            c.pid
            for c in psutil.net_connections() if getattr(c, "laddr", None)
            and getattr(c.laddr, "port", None) == port and c.pid
        }
        for pid in pids:
            self.kill_process_and_children(pid)
        return None
