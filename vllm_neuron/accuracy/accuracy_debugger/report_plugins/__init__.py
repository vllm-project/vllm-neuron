# SPDX-License-Identifier: Apache-2.0
"""Report plugins for accuracy validation steps.

Each plugin handles parsing logs and/or artifacts for a specific validation step,
and generates HTML visualizations for the combined report.
"""

from .base import ReportPlugin, PluginRegistry
from .logit_validation import LogitValidationPlugin
from .kv_analysis import KVAnalysisPlugin
from .task_analysis import TaskAnalysisPlugin
from .tensor_compare import TensorComparePlugin

__all__ = [
    "ReportPlugin",
    "PluginRegistry",
    "LogitValidationPlugin",
    "KVAnalysisPlugin",
    "TensorComparePlugin",
    "TaskAnalysisPlugin",
]
