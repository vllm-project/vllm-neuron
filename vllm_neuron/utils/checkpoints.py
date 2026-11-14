# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import threading
import torch
import concurrent
import queue
import time
from abc import ABC, abstractmethod
from typing import NamedTuple, TYPE_CHECKING

from huggingface_hub import list_repo_files, hf_hub_download
from safetensors import safe_open

from vllm_neuron.utils.weight_loader import (
    get_weight_loader,
    SafetensorsWeightLoader,
)

if TYPE_CHECKING:
    from safetensors import PySafeSlice

logger = logging.getLogger(__name__)

# Retry configuration for the page-cache read loop. A read from a network file
# system (for example FSx or Lustre) can fail with a transient OSError, for
# example "Broken pipe".
_PAGE_CACHE_READ_MAX_ATTEMPTS = 4
_PAGE_CACHE_READ_BASE_BACKOFF_S = 1.0
_PAGE_CACHE_READ_MAX_BACKOFF_S = 8.0


class CheckpointLoadResult(NamedTuple):
    """Result of loading a sharded checkpoint, modeled after torch's load_state_dict.

    Attributes:
        state_dict: Loaded tensors mapped to parameter names.
        missing_keys: Parameter names in mappings whose checkpoint key(s) were not found.
        unexpected_keys: Checkpoint keys not referenced by any mapping value.
    """

    state_dict: dict[str, torch.Tensor]
    missing_keys: list[str]
    unexpected_keys: list[str]


class _CheckpointSource(ABC):
    """Abstract base class for checkpoint sources (local, HuggingFace, S3, etc.)."""

    @abstractmethod
    def get_file_names(self) -> list[str]:
        """Return list of safetensor file keys/paths."""
        pass

    @abstractmethod
    def download_file(self, file_name: str) -> None:
        """Download file if needed (no-op for local sources)."""
        pass

    @abstractmethod
    def get_file_path(self, file_name: str) -> str:
        """Return full file path of file."""
        pass


_SAFETENSORS_INDEX = "model.safetensors.index.json"


def _filter_by_index(file_names: list[str], index_path: str | None) -> list[str]:
    """Keep only the files that ``model.safetensors.index.json`` names.

    Some repositories ship one set of weights in two formats, for example
    ``model-*.safetensors`` and ``consolidated.safetensors``. Upstream vLLM
    loads only the files in the index. Do the same, so the loader does not
    open a duplicate copy of the weights.

    Args:
        file_names: The candidate file names.
        index_path: The local path of the index file, or None if absent.

    Returns:
        The file names in the index, or ``file_names`` if no index exists.

    Examples:
        >>> _filter_by_index(["a.safetensors"], None)
        ['a.safetensors']
    """
    if index_path is None or not os.path.isfile(index_path):
        return file_names
    with open(index_path) as f:
        indexed = set(json.load(f)["weight_map"].values())
    missing = indexed - set(file_names)
    if missing:
        raise FileNotFoundError(
            f"{_SAFETENSORS_INDEX} names files that are not in the checkpoint: "
            f"{sorted(missing)}. Download the missing files, or fix the index."
        )
    return [f for f in file_names if f in indexed]


class _LocalCheckpointSource(_CheckpointSource):
    """
    Checkpoint source for loading files from a local directory.

    Args:
        checkpoint_dir: Path to directory containing checkpoint files.
        file_extension: File extension to filter by (e.g., ".safetensors")
    """

    def __init__(self, checkpoint_dir: str, file_extension: str):
        self._file_dir = checkpoint_dir
        self._file_names = sorted(
            f for f in os.listdir(checkpoint_dir) if f.endswith(file_extension)
        )
        if file_extension == ".safetensors":
            self._file_names = _filter_by_index(
                self._file_names, os.path.join(checkpoint_dir, _SAFETENSORS_INDEX)
            )

    def get_file_names(self) -> list[str]:
        """Return sorted list of filenames in the checkpoint directory."""
        return self._file_names

    def download_file(self, file_name: str) -> None:
        """No-op for local sources."""
        pass

    def get_file_path(self, file_name: str) -> str:
        """Return the absolute path to the specified file."""
        return os.path.join(self._file_dir, file_name)


class _HFCheckpointSource(_CheckpointSource):
    """
    Checkpoint source for HuggingFace Hub models.

    Args:
        model_name: HuggingFace model identifier (e.g., "company_name/custom_model").
        file_extension: File extension to filter by (e.g., ".safetensors").
        cache_dir: Directory for caching downloaded files.
    """

    def __init__(
        self, model_name: str, file_extension: str, cache_dir: str | None = None
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        repo_files = list_repo_files(model_name)
        self._file_names = sorted(
            f for f in repo_files if f.endswith(file_extension) and "/" not in f
        )
        if file_extension == ".safetensors" and _SAFETENSORS_INDEX in repo_files:
            index_path = hf_hub_download(
                model_name, _SAFETENSORS_INDEX, cache_dir=cache_dir
            )
            self._file_names = _filter_by_index(self._file_names, index_path)

    def get_file_names(self) -> list[str]:
        """Return sorted list of filenames in the HuggingFace repository."""
        return self._file_names

    def download_file(self, file_name: str) -> None:
        """Download the file from HuggingFace Hub if not already cached."""
        hf_hub_download(self._model_name, file_name, cache_dir=self._cache_dir)

    def get_file_path(self, file_name: str) -> str:
        """Return local file path."""
        return hf_hub_download(
            self._model_name,
            file_name,
            cache_dir=self._cache_dir,
            local_files_only=True,
        )


def _get_checkpoint_source(
    model_name_or_path: str, file_extension: str, cache_dir: str | None
) -> _CheckpointSource:
    """
    Factory function to get the appropriate _CheckpointSource.

    Args:
        model_name_or_path: HuggingFace model name or local directory path.
        file_extension: File extension to filter by (e.g., ".safetensors", ".bin").
        cache_dir: Cache directory for HuggingFace downloads.
    """
    if os.path.isdir(model_name_or_path):
        return _LocalCheckpointSource(model_name_or_path, file_extension)
    else:
        return _HFCheckpointSource(model_name_or_path, file_extension, cache_dir)


class SafetensorsCheckpoint:
    """
    Manages multi-process loading of tensors from safetensors checkpoint files distributed across multiple files.

    This class supports both local and HuggingFace-backed checkpoints.

    Provides efficient pipelined weight loading in a multi-process environment:
        1. Load checkpoint into OS page cache (each rank loads a portion of files)
        2. Apply weight transformations (sharding, fusion, etc.) via SafetensorsWeightLoader
        3. Transfer processed tensors from CPU to device

    Args:
        model_name_or_path: HuggingFace model name or directory path containing .safetensors files
        cache_dir: Cache directory for HF downloads (only used if model_name_or_path is an HF model)

    Example:
        >>> checkpoint = SafetensorsCheckpoint("meta-llama/Llama-3.1-8B")
        >>> model = Model(...)
        >>>
        >>> result = checkpoint.load_sharded_pipelined(
        ...     rank=rank, world_size=world_size, model=model,
        ...     mappings={}, device=torch.device("neuron")
        ... )
        >>> model.load_state_dict(result.state_dict, strict=False, assign=True)
    """

    def __init__(self, model_name_or_path: str, cache_dir: str | None = None):
        self._source = _get_checkpoint_source(
            model_name_or_path, ".safetensors", cache_dir
        )
        self._safetensor_file_names = self._source.get_file_names()
        assert len(self._safetensor_file_names) > 0, (
            f"No .safetensors files found for {model_name_or_path}"
        )

        # Populated lazily as files from checkpoint are loaded into OS page cache
        self._tensor_name_to_file = {}
        self._open_safetensor_files = {}

    def load_sharded(
        self,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        mappings: dict,
        device: torch.device,
        strict: bool = True,
    ) -> CheckpointLoadResult:
        """
        Simple checkpoint loading without distributed coordination.

        Loads weights sequentially without pipelining. Each rank independently
        loads all checkpoint files and extracts its sharded portion. This is
        simpler but slower than load_sharded_pipelined.

        Use this method for CPU testing or when distributed store coordination
        causes issues.

        Args:
            rank: Tensor parallel rank for this process.
            world_size: Total tensor parallel world size.
            model: Model whose parameters define what to load.
            mappings: Dict mapping parameter names to checkpoint key(s).
            device: Target device for loaded tensors.
            strict: If True, raise RuntimeError when checkpoint keys referenced
                by mappings are not found. If False, skip missing keys and
                return them in the result. Default: True.

        Returns:
            CheckpointLoadResult with state_dict, missing_keys, and unexpected_keys.
        """
        state_dict: dict[str, torch.Tensor] = {}
        missing_keys: list[str] = []

        # Download and open all safetensor files
        for file_name in self._safetensor_file_names:
            self._source.download_file(
                file_name
            )  # Download if needed (no-op for local)
            file_path = self._source.get_file_path(file_name)
            self._open_safetensor_files[file_path] = safe_open(
                file_path, framework="pt", device="cpu"
            )
            for key in self._open_safetensor_files[file_path].keys():
                self._tensor_name_to_file[key] = file_path

        # Collect all checkpoint keys referenced by mappings for unexpected_keys calc
        referenced_checkpoint_keys: set[str] = set()
        for name, _ in model.named_parameters():
            ck = mappings.get(name, name)
            if isinstance(ck, list):
                referenced_checkpoint_keys.update(ck)
            else:
                referenced_checkpoint_keys.add(ck)

        # Process each parameter
        for name, param in model.named_parameters():
            # Get checkpoint key(s) for this parameter
            # If not in mappings, use parameter name as checkpoint key
            checkpoint_keys = mappings.get(name, name)
            checkpoint_keys = (
                checkpoint_keys
                if isinstance(checkpoint_keys, list)
                else [checkpoint_keys]
            )

            # Check if keys exist
            missing = [k for k in checkpoint_keys if k not in self._tensor_name_to_file]
            if missing:
                if strict:
                    raise RuntimeError(
                        f"Checkpoint key(s) not found for parameter '{name}': {missing}. "
                        f"Use strict=False to skip missing keys."
                    )
                missing_keys.append(name)
                continue

            # Load and transform weights
            weight_loader = get_weight_loader(param)
            tensor = weight_loader.load(
                [self._get_slice(k) for k in checkpoint_keys], rank
            )
            state_dict[name] = tensor.to(device)

        unexpected_keys = [
            k for k in self._tensor_name_to_file if k not in referenced_checkpoint_keys
        ]

        return CheckpointLoadResult(state_dict, missing_keys, unexpected_keys)

    def load_sharded_pipelined(
        self,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        mappings: dict,
        device: torch.device,
        dtype_override: dict[str, torch.dtype] | None = None,
        strict: bool = True,
    ) -> CheckpointLoadResult:
        """
        Loads checkpoint in a distributed execution with pipelined data movement:
            - Disk -> OS page cache -> tensor processing -> device transfer

        Orchestrates three concurrent stages:
        1. Background thread loads files from disk into OS page cache
        2. Main thread reads tensors from OS page cache and applies SafetensorsWeightLoader transforms
        3. Background thread transfers processed tensors to device

        Args:
            rank: Tensor parallel rank for this process.
            world_size: Total tensor parallel world size.
            model: Model whose parameters define what to load. Each parameter's
                SafetensorsWeightLoader (if attached) controls how checkpoint
                tensors are transformed (sharding, fusion, etc.).
            mappings: Dict mapping parameter names to checkpoint key(s).
                If parameter name not in mappings, uses parameter name as checkpoint key.
                Value can be a string (single key) or list of strings (multiple keys for fusion).
            device: Target device for loaded tensors.
            dtype_override: Optional dict mapping parameter names to target dtypes.
                e.g. {"layer.weight": torch.bfloat16}. Parameters not in this dict
                are cast to the model parameter's dtype. A warning is logged if the
                checkpoint tensor dtype differs from the target dtype.
            strict: If True, raise RuntimeError when checkpoint keys referenced
                by mappings are not found. If False, skip missing keys and
                return them in the result. Default: True.

        Returns:
            CheckpointLoadResult with state_dict, missing_keys, and unexpected_keys.

        Example:
            >>> # Offline stored checkpoint
            >>> checkpoint = SafetensorsCheckpoint("/path/to/ckpt")
            >>>
            >>> # HuggingFace-backed checkpoint
            >>> checkpoint = SafetensorsCheckpoint("org/model_name")
            >>>
            >>> # mappings can remap keys or specify multiple keys for fusion
            >>> # e.g. {"qkv_weight": ["q_proj.weight", "k_proj.weight", "v_proj.weight"]}
            >>> result = checkpoint.load_sharded_pipelined(
            ...     rank=rank, world_size=world_size, model=model,
            ...     mappings={}, device=torch.device("neuron")
            ... )
            >>> model.load_state_dict(result.state_dict, strict=False, assign=True)
        """

        # Distributed key-value store for communicating which files have been loaded to OS page cache
        cached_files_store: torch.distributed.Store = (
            torch.distributed.distributed_c10d._get_default_store()
        )

        tensor_queue: queue.Queue = (
            queue.Queue()
        )  # Queue for pipelining file -> memory and memory -> HBM tensor transfer
        state_dict: dict[
            str, torch.Tensor
        ] = {}  # State dict that we populate with the rank's weights
        missing_keys: list[str] = []
        shutdown_event = (
            threading.Event()
        )  # Signal for clean shutdown if any thread has an error

        # 1 worker for page cache population, 1 worker for weight transformations, and 2 workers for device transfer
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        futures: list = []
        named_params = list(model.named_parameters())
        num_mapped_params = len(named_params)

        # Collect all checkpoint keys referenced by mappings for unexpected_keys calc
        referenced_checkpoint_keys: set[str] = set()
        for name, _ in named_params:
            ck = mappings.get(name, name)
            if isinstance(ck, list):
                referenced_checkpoint_keys.update(ck)
            else:
                referenced_checkpoint_keys.add(ck)

        try:
            # Start background threads for pipelined execution
            page_cache_future = executor.submit(
                self._load_to_page_cache,
                rank,
                world_size,
                cached_files_store,
                shutdown_event,
            )
            device_future = executor.submit(
                self._load_to_device,
                rank,
                tensor_queue,
                state_dict,
                device,
                num_mapped_params,
                shutdown_event,
            )
            futures.extend([page_cache_future, device_future])

            processed_files: set[str] = set()
            processed_param_names: set[str] = set()

            while len(processed_files) < self.get_num_files():
                # Check if any new files have been read into cache
                for file_name in self._safetensor_file_names:
                    if file_name in processed_files:
                        continue
                    if cached_files_store.check([file_name]):
                        processed_files.add(file_name)

                        # Update tensor-to-file mapping for this file
                        file_path = self._source.get_file_path(file_name)
                        self._open_safetensor_files[file_path] = safe_open(
                            file_path, framework="pt", device="cpu"
                        )
                        for key in self._open_safetensor_files[file_path].keys():
                            self._tensor_name_to_file[key] = file_path

                all_files_are_processed = len(processed_files) == self.get_num_files()

                # Process all parameters with weights that are now in cache
                for name, param in named_params:
                    if name in processed_param_names:
                        continue

                    # Get the checkpoint key names corresponding to parameter
                    # If not in mappings, use parameter name as checkpoint key
                    checkpoint_keys = mappings.get(name, name)
                    checkpoint_keys = (
                        checkpoint_keys
                        if isinstance(checkpoint_keys, list)
                        else [checkpoint_keys]
                    )

                    # Check if all weights needed to create the parameter are ready
                    if not all(
                        key in self._tensor_name_to_file for key in checkpoint_keys
                    ):
                        # All files from a checkpoint have been processed, but weight(s) are missing
                        if all_files_are_processed:
                            missing = [
                                k
                                for k in checkpoint_keys
                                if k not in self._tensor_name_to_file
                            ]
                            if strict:
                                raise RuntimeError(
                                    f"Checkpoint key(s) not found for parameter '{name}': {missing}. "
                                    f"Use strict=False to skip missing keys."
                                )
                            missing_keys.append(name)
                            # Send skip sentinel so _load_to_device count stays correct
                            tensor_queue.put((name, None))
                            processed_param_names.add(name)
                        continue

                    # Weights are ready to be processed
                    target_dtype = (dtype_override or {}).get(name, param.dtype)
                    weight_loader = get_weight_loader(param)
                    future = executor.submit(
                        self._process_param,
                        name,
                        checkpoint_keys,
                        weight_loader,
                        rank,
                        target_dtype,
                        tensor_queue,
                    )
                    futures.append(future)
                    processed_param_names.add(name)

                # Check if any futures raised exceptions (silent exceptions in futures could lead to hanging)
                for future in futures:
                    if future.done() and future.exception() is not None:
                        raise future.exception()

                # Sleep to minimize contention
                time.sleep(0.001)

            # Wait for all futures to complete or for one to raise an exception
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            )
            for future in done:
                # Raises an exception if there was one
                future.result()

        except Exception as e:
            logger.error(f"Exception during load_shard_pipelined(): {e}")
            shutdown_event.set()
            tensor_queue.put(
                (None, None)
            )  # Unblock _load_to_device if waiting on queue
            raise e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        unexpected_keys = [
            k for k in self._tensor_name_to_file if k not in referenced_checkpoint_keys
        ]

        return CheckpointLoadResult(state_dict, missing_keys, unexpected_keys)

    def _load_to_page_cache(
        self,
        rank: int,
        world_size: int,
        cached_files_store: torch.distributed.Store,
        shutdown_event: threading.Event,
    ) -> None:
        """
        Loads this rank's portion of checkpoint files into the OS page cache.

        Preloading into the page cache significantly reduces latency when reading non-contiguous
        tensor shards, as the data is already in RAM rather than requiring strided disk I/O which
        is very slow. Files are distributed across ranks using round-robin (idx % world_size == rank).

        Args:
            rank: Current process rank in distributed setup (0-indexed). Determines which
                subset of files this process will cache.
            world_size: Total number of processes in the distributed job. Files are
                distributed across ranks via round-robin.
            cached_files_store: Distributed key-value store (e.g., torch.distributed.Store)
                to track which files have been successfully cached. Keys are file names,
                values are local file paths.
            shutdown_event: Event to signal early termination if an error occurs elsewhere.
        """
        import time

        start_time = time.perf_counter()

        # Used for logging the files read by this rank
        cached_files = []

        try:
            # Load all files into OS page cache
            for idx, file_name in enumerate(self._source.get_file_names()):
                if shutdown_event.is_set():
                    return

                if idx % world_size == rank:
                    # Download the file if needed, then get the local path.
                    self._source.download_file(file_name)
                    file_path = self._source.get_file_path(file_name)

                    # Read the file into the OS page cache with retries.
                    # A shutdown signal during a retry stops this rank early.
                    if not self._read_file_into_page_cache(file_path, shutdown_event):
                        return

                    # Add to distributed store so other threads/processes know this file is ready
                    cached_files_store.add(file_name, 1)
                    cached_files.append(file_name)

        except Exception as e:
            logger.error(f"Error in _load_to_page_cache(): {e}")
            raise

        logger.info(
            f"Finished populating page cache files {cached_files}: {time.perf_counter() - start_time}"
        )

    def _read_file_into_page_cache(
        self, file_path: str, shutdown_event: threading.Event
    ) -> bool:
        """Read a checkpoint file into the OS page cache with retry.

        Retry the local read of the file with exponential backoff. A network
        file system (for example FSx or Lustre) can drop a read with an OSError,
        for example "Broken pipe". A retry reads the file again, which is safe.

        Args:
            file_path: Local path of the checkpoint file to read into the page
                cache.
            shutdown_event: Event that signals an early stop. The method stops
                between retry attempts if the event is set.

        Returns:
            True if the read succeeded. False if a shutdown stopped the method
            during a retry backoff.

        Raises:
            OSError: The read did not succeed after the maximum number of attempts.
        """
        chunk_size = 4 * 1024 * 1024  # 4 MB
        backoff_s = _PAGE_CACHE_READ_BASE_BACKOFF_S
        for attempt in range(1, _PAGE_CACHE_READ_MAX_ATTEMPTS + 1):
            try:
                # Read through the file to load it into the OS page cache.
                with open(file_path, "rb") as f:
                    while chunk := f.read(chunk_size):
                        pass
                return True
            except OSError as e:
                if attempt >= _PAGE_CACHE_READ_MAX_ATTEMPTS:
                    logger.error(
                        "Failed to read file %s into the page cache after %d attempts: %s",
                        file_path,
                        attempt,
                        e,
                    )
                    raise
                logger.warning(
                    "Cannot read file %s into the page cache (attempt %d/%d): %s. "
                    "Retry in %.1f seconds.",
                    file_path,
                    attempt,
                    _PAGE_CACHE_READ_MAX_ATTEMPTS,
                    e,
                    backoff_s,
                )
                # Stop early if another thread signals a shutdown.
                if shutdown_event.wait(timeout=backoff_s):
                    return False
                backoff_s = min(_PAGE_CACHE_READ_MAX_BACKOFF_S, backoff_s * 2)

    def _process_param(
        self,
        name: str,
        checkpoint_keys: list[str],
        weight_loader: SafetensorsWeightLoader,
        rank: int,
        target_dtype: torch.dtype,
        tensor_queue: queue.Queue,
    ) -> None:
        """Load checkpoint slices for a parameter and add to queue for device transfer.

        Retrieves tensor slices from checkpoint files, applies weight loader
        transformations (sharding, fusion, etc.), and places the result in
        the tensor queue for CPU -> device transfer.

        Args:
            name: Parameter name in the model's state dict.
            checkpoint_keys: Checkpoint tensor key(s) to load. Multiple keys
                may be used (e.g. fused QKV)
            weight_loader: Transforms checkpoint slices into the final tensor.
            rank: Rank for sharding weight.
            target_dtype: Target dtype for the tensor.
            tensor_queue: Queue for passing tensors to device transfer thread.
        """
        torch.set_num_threads(1)  # Avoid OpenMP conflicts in thread pool
        tensor = weight_loader.load([self._get_slice(k) for k in checkpoint_keys], rank)
        if tensor.dtype != target_dtype:
            logger.warning(
                f"Mismatch between parameter {name} defined dtype {target_dtype} and weight loader-generated "
                f"dtype {tensor.dtype}, casting to {target_dtype}."
            )
            tensor = tensor.to(target_dtype)
        tensor_queue.put((name, tensor))

    def _load_to_device(
        self,
        rank: int,
        tensor_queue: queue.Queue,
        state_dict: dict[str, torch.Tensor],
        device: torch.device,
        num_params: int,
        shutdown_event: threading.Event,
    ) -> None:
        """
        Transfers tensors from CPU memory to device HBM.

        This method enables pipelined checkpoint loading by decoupling CPU-side tensor preparation
        (reading from disk, sharding, padding) from device transfer. Continuously pulls tensors from
        a queue and transfers them to device memory (HBM), maximizing hardware utilization and
        minimizing idle time.

        Blocks on the queue when empty and terminates after processing num_params tensors,
        or when a shutdown signal (None, None) is received.

        Args:
            rank: Current process rank in distributed setup. Used for error logging.
            tensor_queue: Thread-safe queue containing (name, tensor) tuples to transfer.
            state_dict: Dictionary to populate with device tensors.
            device: Target device for tensor transfer.
            num_params: Number of tensors to process before terminating.
            shutdown_event: Event to signal early termination if an error occurs elsewhere.
        """
        try:
            tensor_count = 0
            while tensor_count < num_params:
                if shutdown_event.is_set():
                    return

                name, cpu_tensor = tensor_queue.get()

                if name is None:  # Shutdown signal
                    return

                # Skip sentinel for missing keys (strict=False)
                if cpu_tensor is None:
                    tensor_count += 1
                    continue

                # Load tensor to device
                device_tensor = cpu_tensor.to(device)

                # Store in state dict
                state_dict[name] = device_tensor
                tensor_count += 1

        except Exception as e:
            logger.error(f"[Rank {rank}] Error in device_load_checkpoint: {e}")
            raise

    def _ensure_indexed(self):
        """Open all safetensor files and build the tensor name index without loading data."""
        for file_name in self._safetensor_file_names:
            self._source.download_file(file_name)
            file_path = self._source.get_file_path(file_name)
            if file_path not in self._open_safetensor_files:
                self._open_safetensor_files[file_path] = safe_open(
                    file_path, framework="pt", device="cpu"
                )
                for key in self._open_safetensor_files[file_path].keys():
                    self._tensor_name_to_file[key] = file_path

    def get_tensor_names(self) -> set[str]:
        """Return the set of tensor names available in the checkpoint."""
        self._ensure_indexed()
        return set(self._tensor_name_to_file.keys())

    def get_tensor(self, name: str) -> torch.Tensor | None:
        """Return one full tensor from the checkpoint, or None if it is absent.

        Use this method for a small tensor that the model does not shard, for
        example a token id map. The tensor stays on the CPU.

        Args:
            name: The tensor name in the safetensors files.

        Returns:
            The tensor, or None if no file contains ``name``.

        Examples:
            >>> checkpoint = SafetensorsCheckpoint("/path/to/checkpoint")
            >>> d2t = checkpoint.get_tensor("d2t")
        """
        self._ensure_indexed()
        file_path = self._tensor_name_to_file.get(name)
        if file_path is None:
            return None
        return self._open_safetensor_files[file_path].get_tensor(name)

    def get_num_files(self) -> int:
        """
        Returns the total number of safetensor files found in the checkpoint when initialized.

        Returns:
            int: Total number of safetensor (.safetensors) files in the checkpoint directory.

        Examples:
            Basic usage to check checkpoint size:

            >>> checkpoint = SafetensorsCheckpoint("/path/to/checkpoint")
            >>> num_files = checkpoint.get_num_files()
        """
        return len(self._safetensor_file_names)

    def _get_slice(self, name: str) -> "PySafeSlice":
        """Retrieve a slice object for a named tensor in the checkpoint.

        Looks up the safetensor file containing the tensor name and returns a slice
        object that can be used to access portions of the tensor without loading the
        entire tensor into memory.

        Args:
            name: The name of the tensor to retrieve a slice for.

        Returns:
            PySafeSlice: A slice object for the named tensor that supports
                efficient partial loading.
        """
        file_with_tensor = self._tensor_name_to_file[name]
        open_safetensor_file = self._open_safetensor_files[file_with_tensor]
        return open_safetensor_file.get_slice(name)
