# SPDX-License-Identifier: Apache-2.0
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class BenchmarkFSxWriter:

    def __init__(self):
        self.fsx_base_path = os.environ.get('FSX_TEAM_SHARED_RW')
        if not self.fsx_base_path:
            raise ValueError("FSX_TEAM_SHARED_RW environment variable not set")
        self.fsx_base_path = Path(self.fsx_base_path)

    def write_results_to_fsx(self,
                             local_path: Path,
                             model_name: str,
                             config_name: str,
                             test_name: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Write benchmark results to FSx with structured organization
        Args:
            local_path: Path to local results directory
            model_name: Name of the model (e.g., llama33_70b)
            config_name: Configuration name (e.g., base_1k256_b4_tp32)
            test_name: Name of the test
            metadata: Additional metadata to attach
        Returns:
            FSx path where results were written
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create structured FSx paths (matching Kaizen's expected structure)
        benchmark_path = self.fsx_base_path / "benchmarks" / model_name / config_name / "runs" / timestamp

        # Create directory if it doesn't exist
        benchmark_path.mkdir(parents=True, exist_ok=True)

        # Write metadata to a separate file
        if metadata:
            metadata_path = benchmark_path / "metadata.json"
            with metadata_path.open('w') as f:
                json.dump(metadata, f, indent=2)

        # Copy relevant files
        uploaded_files = []
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Only copy extended summary files
                if "extended_summary_" not in file_path.name:
                    continue

                # Create destination path
                dest_path = benchmark_path / file_path.name

                # Copy file
                shutil.copy2(file_path, dest_path)
                uploaded_files.append(str(dest_path))

        return str(benchmark_path)
