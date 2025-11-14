# SPDX-License-Identifier: Apache-2.0
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    from .model_registry import ModelRegistry
except ImportError as e:
    ModelRegistry = None
    logger.warning(
        "Failed to import ModelRegistry - HF <-> FSX mappings will not work! "
        f"Import error: {e}")

_FSX_SUBDIR = "neuronx-distributed/inference"


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "")
    return v if v else default


def resolve_model_dir(logical: str,
                      require_fsx: bool = False,
                      use_fsx_only: bool = False) -> Tuple[str, bool]:
    """
    Resolve model identifier bidirectionally based on FSX availability.
    Args:
        logical: Model identifier (absolute path, HF ID, FSx path, or custom model path)
        require_fsx: If True, raise error when model not found on FSx
        use_fsx_only: If True, skip SSD caching even if available
    Returns:
        Tuple of (resolved_path_or_id, used_ssd_copy)
    """
    original_logical = logical
    logger.info(f"Resolving model: '{original_logical}'")

    # Try to resolve through ModelRegistry if available
    if ModelRegistry is not None:
        resolved = ModelRegistry.resolve_model(logical)
        if resolved != logical:
            logger.info(f"Registry resolved: '{logical}' -> '{resolved}'")
        logical = resolved
    else:
        logger.info("ModelRegistry not available, using model as-is")

    # 1) Handle absolute paths, we return them as is
    if os.path.isabs(logical):
        logger.info(f"Using absolute path: '{logical}'")
        return logical, False

    # 2) Check FSx location
    fsx_root = _env("FSX_SHARED_RO") or _env("FSX_RO")

    if not fsx_root:
        if require_fsx:
            logger.error(
                f"FSx not configured but require_fsx=True for model: '{original_logical}'"
            )
            raise FileNotFoundError(
                f"FSx not configured (FSX_SHARED_RO/FSX_RO not set) "
                f"but require_fsx=True for model: {original_logical}")
        logger.info(f"No FSx available, returning model ID: '{logical}'")
        return logical, False

    fsx_path = Path(fsx_root) / _FSX_SUBDIR / logical
    logger.info(f"Checking FSx path: {fsx_path}")

    if not fsx_path.exists():
        if require_fsx:
            logger.error(f"Model not found on FSx: {fsx_path}")
            raise FileNotFoundError(f"Model not found on FSx: {fsx_path} "
                                    f"(original: {original_logical})")
        logger.info(
            f"Model not found on FSx, returning as model ID: '{logical}'")
        return logical, False

    logger.info(f"Model found on FSx: {fsx_path}")

    # If use_fsx_only is set, return FSx path directly
    if use_fsx_only:
        logger.info(
            f"Returning FSx path directly (use_fsx_only=True): '{fsx_path}'")
        return str(fsx_path), False

    ssd_root = _env("SSD_RW")

    if not ssd_root:
        logger.info(f"No SSD configured, using FSx path: '{fsx_path}'")
        return str(fsx_path), False

    ssd_root_path = Path(ssd_root)

    if not ssd_root_path.exists() or not os.access(ssd_root, os.W_OK):
        logger.warning(
            f"SSD not writable ({ssd_root}), using FSx path: '{fsx_path}'")
        return str(fsx_path), False

    # Prepare SSD destination
    dst = ssd_root_path / _FSX_SUBDIR / logical
    done_marker = dst / ".copy_complete"

    if done_marker.exists():
        logger.info(f"Using cached SSD copy: '{dst}'")
        return str(dst), True

    # Need to copy to SSD
    logger.info(f"Starting copy to SSD: '{fsx_path}' -> '{dst}'")

    try:
        if dst.exists():
            logger.info(f"Removing partial copy at '{dst}'")
            shutil.rmtree(dst, ignore_errors=True)

        dst.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Copying model files to SSD...")
        shutil.copytree(fsx_path,
                        dst,
                        dirs_exist_ok=False,
                        copy_function=shutil.copy2)

        done_marker.write_text("ok", encoding="utf-8")
        logger.info(f"Successfully cached model to SSD: '{dst}'")

        return str(dst), True

    except Exception as e:
        logger.error(f"Failed to copy model to SSD: {e}")
        logger.info(f"Falling back to FSx path: '{fsx_path}'")
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        return str(fsx_path), False
