# SPDX-License-Identifier: Apache-2.0
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ModelRegistry:
    # Registry for HF model ID <-> FSx path mappings

    _mappings: Dict[str, str] = {}
    _reverse_mappings: Dict[str, str] = {}
    _loaded: bool = False

    @classmethod
    def load_mappings(cls, path: Optional[str] = None) -> None:
        # Load mappings from YAML file
        if cls._loaded:
            return

        # Use env var or default location
        if not path:
            path = os.environ.get("VLLM_MODEL_MAPPINGS", "fsx_hf_mapping.yaml")

        mapping_file = Path(path)

        # Try standard locations if default doesn't exist
        if not mapping_file.exists():
            for try_path in [
                    Path(__file__).parent / "fsx_hf_mapping.yaml",
                    Path.cwd() / "fsx_hf_mapping.yaml",
            ]:
                if try_path.exists():
                    mapping_file = try_path
                    break

        if mapping_file.exists():
            with open(mapping_file) as f:
                data = yaml.safe_load(f)
                cls._mappings = data.get("mappings", {})
                cls._reverse_mappings = {
                    v: k
                    for k, v in cls._mappings.items()
                }
                logger.info(
                    f"Loaded {len(cls._mappings)} model mappings from {mapping_file}"
                )
        else:
            logger.info("No model mappings found. Models will be used as-is.")
            cls._mappings = {}
            cls._reverse_mappings = {}

        cls._loaded = True

    @classmethod
    def is_fsx_available(cls) -> bool:
        # Check FSX availability
        fsx_root = os.environ.get("FSX_SHARED_RO") or os.environ.get("FSX_RO")
        if not fsx_root:
            return False
        fsx_path = Path(fsx_root) / "neuronx-distributed" / "inference"
        exists = fsx_path.exists()
        if exists:
            logger.info(f"FSx is available at: {fsx_path}")
        return exists

    @classmethod
    def resolve_model(cls, model_id: str) -> str:
        """
        Resolve model identifier bidirectionally based on FSX availability.
    
        Resolution behavior (5 cases):
        
        1. HuggingFace ID + FSX available
        Input: 'meta-llama/Llama-3.1-8B'
        Consults mapping -> Returns FSX path
        
        2. HuggingFace ID + FSX unavailable
        Input: 'meta-llama/Llama-3.1-8B'
        Consults FSX→HF mapping (not found) -> Returns input as-is
        
        3. FSX path + FSX available
        Input: 'llama-3.1/llama-3.1-8b'
        Consults HF→FSX mapping (not found) -> Returns input as-is
        
        4. FSX path + FSX unavailable
        Input: 'llama-3.1/llama-3.1-8b'
        Consults FSX→HF mapping -> Returns HuggingFace ID
        
        5. Unmapped input (absolute paths, unknown models, etc.)
        Input: '/custom/model' 
        No mapping found -> Returns input as-is
        
        Args:
            model_id: Either a HuggingFace model ID or FSX relative path
        
        Returns:
            Resolved model identifier appropriate for current environment
        """
        # Ensure mappings are loaded
        if not cls._loaded:
            cls.load_mappings()

        original = model_id

        if cls.is_fsx_available():
            # In FSx environment - check if we have a mapping
            if model_id in cls._mappings:
                resolved = cls._mappings[model_id]
                logger.info(
                    f"Model resolved (FSx environment): '{original}' -> '{resolved}'"
                )
                return resolved
            else:
                # No mapping, use as-is (could be FSx-only path or unknown model)
                logger.info(
                    f"No mapping found, using as-is in FSx environment: '{original}'"
                )
                return model_id
        else:
            # Not in FSx environment - check for reverse mapping
            if model_id in cls._reverse_mappings:
                resolved = cls._reverse_mappings[model_id]
                logger.info(
                    f"Model resolved (non-FSx environment): '{original}' -> '{resolved}'"
                )
                return resolved
            else:
                # No reverse mapping, use as-is (could be HF ID or will fail if FSx-only)
                logger.info(
                    f"No mapping found, using as-is in non-FSx environment: '{original}'"
                )
                return model_id


# to automatically load on import
ModelRegistry.load_mappings()
