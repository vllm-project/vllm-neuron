# SPDX-License-Identifier: Apache-2.0
"""Base class and registry for report plugins."""

import html as _html
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class PluginResult:
    """Result from a plugin's parse/build operations."""

    passed: Optional[bool] = None
    summary: str = ""
    text_summary: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    html: str = ""


class ReportPlugin(ABC):
    """Base class for report step plugins.

    Each plugin handles one validation step (e.g., logit validation, KV analysis).
    Plugins can parse log files and/or load artifacts (.pt files, HTML) to generate
    visualizations for the combined report.
    """

    # Plugin metadata - override in subclasses
    name: str = "base"
    display_name: str = "Base Plugin"
    step_index: int = 0
    guide_text: str = ""

    def __init__(
        self, prompt_dir: Optional[str] = None, capture_dir: Optional[str] = None
    ):
        self.prompt_dir = prompt_dir
        self.capture_dir = capture_dir
        self._data: Optional[Dict] = None

    @property
    def log_filename(self) -> str:
        """Default log filename for this step."""
        return f"step{self.step_index}_{self.name}.txt"

    def get_log_path(self) -> Optional[str]:
        """Get full path to log file."""
        if not self.prompt_dir:
            return None
        path = os.path.join(self.prompt_dir, self.log_filename)
        return path if os.path.isfile(path) else None

    def get_capture_subdir(self, subdir: str) -> Optional[str]:
        """Get path to a capture subdirectory."""
        if not self.capture_dir:
            return None
        path = os.path.join(self.capture_dir, subdir)
        return path if os.path.isdir(path) else None

    @abstractmethod
    def parse_log(self, log_path: str) -> Optional[Dict]:
        """Parse the step's log file into structured data.

        Args:
            log_path: Path to the log file

        Returns:
            Parsed data dict, or None if parsing fails
        """
        pass

    def load_artifacts(self, artifact_dir: str) -> Optional[Dict]:
        """Load and process artifacts (optional).

        Override this to load .pt files or other artifacts for visualization.

        Args:
            artifact_dir: Path to artifacts directory

        Returns:
            Processed artifact data, or None
        """
        return None

    @abstractmethod
    def build_html(self, data: Dict) -> str:
        """Build HTML visualization from parsed data.

        Args:
            data: Parsed data from parse_log() and/or load_artifacts()

        Returns:
            HTML string for this section
        """
        pass

    @abstractmethod
    def build_text_summary(self, data: Dict) -> str:
        """Build a plain-text summary for logging and agent consumption.

        Args:
            data: Parsed data from parse_log() and/or load_artifacts()

        Returns:
            Multi-line text summary with key metrics
        """
        pass

    def check_status(self, data: Dict) -> tuple[bool, str]:
        """Check pass/fail status from parsed data.

        Args:
            data: Parsed data

        Returns:
            Tuple of (passed: bool, detail: str)
        """
        return True, "OK"

    def run(
        self, log_path: Optional[str] = None, artifact_dir: Optional[str] = None
    ) -> PluginResult:
        """Run the full plugin pipeline: parse → load artifacts → build HTML.

        Args:
            log_path: Override log path (uses default if None)
            artifact_dir: Override artifact dir (uses capture_dir if None)

        Returns:
            PluginResult with parsed data and generated HTML
        """
        log_path = log_path or self.get_log_path()
        artifact_dir = artifact_dir or self.capture_dir

        result = PluginResult()

        # Parse log
        if log_path and os.path.isfile(log_path):
            result.data = self.parse_log(log_path) or {}

        # Load artifacts
        if artifact_dir:
            artifacts = self.load_artifacts(artifact_dir)
            if artifacts:
                result.data.update(artifacts)

        if not result.data:
            result.html = f"<p>No {self.display_name} data available.</p>"
            return result

        # Check status
        result.passed, result.summary = self.check_status(result.data)

        # Build HTML
        result.html = self.build_html(result.data)

        # Build text summary
        result.text_summary = self.build_text_summary(result.data)

        return result

    def wrap_with_guide(self, html: str) -> str:
        """Wrap HTML content with the guide text and reproduction info."""
        parts = []

        # Reproduction info
        log_path = self.get_log_path()
        if log_path and os.path.isfile(log_path):
            with open(log_path) as f:
                first_line = f.readline().strip()
            if first_line.startswith(">>>"):
                cmd = first_line[4:]
                capture_dir = self.capture_dir or ""
                parts.append(
                    '<details class="guide"><summary>Reproduce / Output</summary>'
                    f"<p><b>Command:</b></p><pre>{_html.escape(cmd)}</pre>"
                    f"<p><b>Log:</b> <code>{_html.escape(log_path)}</code></p>"
                )
                if capture_dir:
                    parts.append(
                        f"<p><b>Artifacts:</b> <code>{_html.escape(capture_dir)}</code></p>"
                    )
                parts.append("</details>")

        if self.guide_text:
            parts.append(
                f'<details class="guide"><summary>How to read this tab</summary>{self.guide_text}</details>'
            )

        parts.append(html)
        return "\n".join(parts)

    # Utility methods
    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from text."""
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    @staticmethod
    def extract_body(html: str) -> str:
        """Extract content between <body> tags, stripping plotly bundle."""
        m = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL)
        body = m.group(1) if m else html
        body = re.sub(
            r"<script[^>]*>\s*window\.PlotlyConfig\s*=.*?</script>\s*"
            r"<script[^>]*>/\*\*\s*\*\s*plotly\.js.*?</script>",
            "",
            body,
            flags=re.DOTALL,
        )
        return body


class PluginRegistry:
    """Registry for report plugins."""

    _plugins: Dict[str, Type[ReportPlugin]] = {}

    @classmethod
    def register(cls, plugin_class: Type[ReportPlugin]) -> Type[ReportPlugin]:
        """Register a plugin class. Can be used as decorator."""
        cls._plugins[plugin_class.name] = plugin_class
        return plugin_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[ReportPlugin]]:
        """Get a plugin class by name."""
        return cls._plugins.get(name)

    @classmethod
    def all(cls) -> List[Type[ReportPlugin]]:
        """Get all registered plugins sorted by step_index."""
        return sorted(cls._plugins.values(), key=lambda p: p.step_index)

    @classmethod
    def names(cls) -> List[str]:
        """Get all registered plugin names."""
        return list(cls._plugins.keys())
