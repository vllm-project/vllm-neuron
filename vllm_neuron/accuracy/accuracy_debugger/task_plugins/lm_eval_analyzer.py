# SPDX-License-Identifier: Apache-2.0
"""
Module for analyzing task-level accuracy results in Neuron models.
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Deviation:
    """Class representing a deviation between reference and target model outputs."""

    doc_id: int
    prompt: str
    ref_results: Dict
    target_results: Dict
    worse_metrics: List[str]


@dataclass
class Match:
    """Class representing a match between reference and target model outputs."""

    doc_id: int
    prompt: str
    ref_results: Dict
    target_results: Dict


class LmEvalAnalyzer:
    """Analyzer for task-level accuracy results in Neuron models."""

    def __init__(
        self,
        ref_result_path: Path = None,
        target_results_path: Path = None,
        output_path: Path = None,
    ):
        """Initialize the task analyzer with reference and target results."""
        self.ref_result_path = Path(ref_result_path) if ref_result_path else None
        self.target_results_path = (
            Path(target_results_path) if target_results_path else None
        )
        self.output_path = Path(output_path) if output_path else None
        self.task_deviation_data = {}
        self.metric_list = []
        self.available_tasks = []

        if self.output_path and self.output_path.exists():
            self._load_results()

    def _load_results(self):
        """Load and parse results from output file."""
        if not self.output_path or not self.output_path.exists():
            return

        try:
            with self.output_path.open("r") as f:
                results = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load results from %s: %s", self.output_path, e)
            return

        # Convert loaded results back to internal format
        task_deviation_data = {}
        for task_name, task_data in results.items():
            if task_name in ["summary_metrics", "ref_metrics", "target_metrics"]:
                continue

            matches = []
            deviations = []

            # Process matching entries
            for doc_id, match_data in task_data.get("matching", {}).items():
                matches.append((int(doc_id), "", match_data, match_data))

            # Process deviating entries
            for doc_id, dev_data in task_data.get("deviating", {}).items():
                deviations.append(
                    Deviation(
                        int(doc_id),
                        "",
                        dev_data,
                        dev_data,
                        dev_data.get("worse_metrics", []),
                    )
                )

            task_metrics = {
                "ref_metrics": task_data.get("ref_metrics", {}),
                "target_metrics": task_data.get("target_metrics", {}),
            }

            task_deviation_data[task_name] = (matches, deviations, task_metrics)

        self.task_deviation_data = task_deviation_data

    def resolve_results_dir(
        self, input_task_results: "str | dict", output_dir: "str | Path"
    ) -> Path:
        """Resolve pre-computed eval results to a directory of lm_eval outputs.

        Accepts either a path to an existing results directory (e.g. an lm_eval
        ``--output_path``) or a results dict, which is written out under
        ``<output_dir>/eval_results`` and returned. This keeps the results
        wrangling with the analyzer that understands the on-disk layout, so the
        ``run_task_analysis`` API stays eval-runner agnostic.
        """
        if isinstance(input_task_results, str):
            return Path(input_task_results).expanduser()
        if isinstance(input_task_results, dict):
            results_dir = Path(output_dir).expanduser() / "eval_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "results.json").write_text(
                json.dumps(input_task_results, indent=2)
            )
            return results_dir
        raise ValueError(
            "input_task_results must be a results dir path or a results dict"
        )

    def get_results_file(self, directory: Path, task_name: str = None) -> Path:
        """Find and return the path to a results_*.json file in the specified directory."""
        results_files = list(directory.glob("results_*.json"))
        if not results_files:
            results_files = list(directory.rglob("results_*.json"))
        if not results_files:
            raise FileNotFoundError(f"No results_*.json files found in {directory}")

        if task_name:
            for results_file in results_files:
                try:
                    with results_file.open("r") as f:
                        data = json.load(f)
                        results = data.get("results", {})
                        if task_name in results or any(
                            key.startswith(task_name + "-") for key in results.keys()
                        ):
                            return results_file
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        "Failed to parse results file %s: %s", results_file, e
                    )
                    continue
            raise FileNotFoundError(f"No results file found for task: {task_name}")

        return results_files[0]

    def get_task_name_from_path(self, filepath: Path) -> str:
        """Extract task name from a sample file path."""
        base_name = filepath.name
        timestamp_pattern = r"_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+\.jsonl$"
        without_timestamp = re.sub(timestamp_pattern, "", base_name)
        task_name = without_timestamp.replace("samples_", "")
        return task_name

    def get_matching_sample_filepaths(
        self, ref_dir: Path, target_dir: Path
    ) -> List[Tuple[Path, Path]]:
        """Find matching pairs of sample files between reference and target directories."""
        matching_pairs = []

        ref_samples = list(ref_dir.rglob("samples_*.jsonl"))
        target_samples = list(target_dir.rglob("samples_*.jsonl"))

        ref_samples_dict = {self.get_task_name_from_path(f): f for f in ref_samples}
        target_samples_dict = {
            self.get_task_name_from_path(f): f for f in target_samples
        }

        for subtask in ref_samples_dict:
            if subtask in target_samples_dict:
                matching_pairs.append(
                    (ref_samples_dict[subtask], target_samples_dict[subtask])
                )

        return matching_pairs

    def get_available_tasks(self, ref_dir: Path, target_dir: Path) -> List[str]:
        """Get list of available task names."""
        ref_samples = list(ref_dir.rglob("samples_*.jsonl"))
        target_samples = list(target_dir.rglob("samples_*.jsonl"))

        ref_tasks = {self.get_task_name_from_path(f) for f in ref_samples}
        target_tasks = {self.get_task_name_from_path(f) for f in target_samples}

        return list(ref_tasks & target_tasks)

    def load_metric_list(self, results_path: Path, task_name: str) -> List[Dict]:
        """Load metric list from results_*.json file for the specified task."""
        try:
            with results_path.open("r") as f:
                results = json.load(f)

            configs = results.get("configs", {})
            group_map = results.get("group_subtasks", {})

            def get_metric_list(task: str) -> List[Dict]:
                if task in configs and "metric_list" in configs[task]:
                    return configs[task]["metric_list"]
                elif task in group_map:
                    metric_lists = []
                    for subtask in group_map[task]:
                        sublist = get_metric_list(subtask)
                        if sublist:
                            metric_lists.extend(sublist)
                    return metric_lists
                return []

            return get_metric_list(task_name)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("Failed to load metric list from %s: %s", results_path, e)
            return []

    def load_summary_metrics(self, results_path: Path) -> Dict[str, float]:
        """Load summary metrics from results_*.json file."""

        def is_valid_metric(key: str, value: Any) -> bool:
            return isinstance(value, (int, float)) and value != 0 and key != "alias"

        def extract_metrics(
            task_name: str, metrics_dict: Dict[str, Any]
        ) -> Dict[str, float]:
            return {
                f"{task_name}-{k}": v
                for k, v in metrics_dict.items()
                if is_valid_metric(k, v)
            }

        def recurse(
            task: str,
            results: Dict[str, Any],
            group_map: Dict[str, list],
            prefix: str = "",
        ) -> Dict[str, float]:
            full_name = f"{prefix}-{task}" if prefix else task
            if task in group_map:
                if not group_map[task]:
                    if task in results:
                        return extract_metrics(full_name, results[task])
                    else:
                        return {}
                else:
                    collected = {}
                    for subtask in group_map[task]:
                        collected.update(
                            recurse(subtask, results, group_map, full_name)
                        )
                    return collected
            elif task in results:
                return extract_metrics(full_name, results[task])
            else:
                return {}

        try:
            with results_path.open("r") as f:
                results_data = json.load(f)

            results = results_data.get("results", {})
            group_map = results_data.get("group_subtasks", {})
            all_metrics = {}

            processed_tasks = set()

            for group_task in group_map:
                all_metrics.update(recurse(group_task, results, group_map))
                processed_tasks.add(group_task)

            for task in results:
                if task not in processed_tasks:
                    all_metrics.update(recurse(task, results, group_map))

            return all_metrics

        except Exception as e:
            logger.error("Error loading summary metrics from %s: %s", results_path, e)
            return {}

    def extract_scores(self, results_dir: Path) -> Dict[str, float]:
        """Read flat ``{metric_key: value}`` scores from lm_eval result files.

        Scans ``results_*.json`` under *results_dir* and flattens every numeric
        metric in each task's ``results[task]`` block (the same raw keys
        thresholds use, e.g. ``"exact_match,flexible-extract"``). Group/aggregate
        tasks are included alongside their subtasks. Missing/unreadable files
        yield an empty dict — pass/fail then fails any configured threshold,
        surfacing the problem.

        Distinct from :meth:`load_summary_metrics`, which prefixes keys with the
        task name and drops zero values; this returns the raw threshold keys.
        """
        scores: Dict[str, float] = {}
        results_files = sorted(Path(results_dir).rglob("results_*.json"))
        for rf in results_files:
            try:
                data = json.loads(rf.read_text())
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not read results file %s: %s", rf, e)
                continue
            results = data.get("results", {})
            for _task, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                for key, value in metrics.items():
                    if key != "alias" and isinstance(value, (int, float)):
                        scores[key] = value
        return scores

    def get_prompt_text(self, doc_data: Dict) -> str:
        """Extract prompt/problem text from doc data."""
        prompt_fields = ["prompt", "problem", "question", "text", "input"]

        for field in prompt_fields:
            if field in doc_data:
                return doc_data[field]

        return str(doc_data)

    def load_samples_data(self, samples_path: Path) -> Dict[int, Dict]:
        """Load evaluation samples/test cases from a JSONL file."""
        samples_data = {}
        with samples_path.open("r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    doc_id = entry.get("doc_id")
                    if doc_id is not None:
                        samples_data[doc_id] = entry
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Error parsing JSONL line in %s: %s", samples_path, e
                    )
                    pass
        return samples_data

    def _load_metric_list(self, target_dir: Path) -> List[Dict]:
        """Load metric list from target results."""
        try:
            target_results_path = self.get_results_file(target_dir)
            with target_results_path.open("r") as f:
                results_data = json.load(f)
                configs = results_data.get("configs", {})
                for config in configs.values():
                    if "metric_list" in config:
                        return config["metric_list"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Results file not found for metric list loading: %s", e)
            pass
        return [{"metric": "exact_match", "higher_is_better": True}]

    def _process_task_samples(
        self, ref_data: Dict, target_data: Dict, metric_list: List[Dict]
    ) -> Tuple[List[Match], List[Deviation]]:
        """Process samples to identify matches and deviations."""
        matches = []
        deviations = []

        for doc_id in ref_data:
            if doc_id not in target_data:
                continue

            ref_result = ref_data[doc_id]
            target_result = target_data[doc_id]
            prompt = self.get_prompt_text(ref_result["doc"])

            all_match, worse_metrics = self.compare_results(
                ref_result, target_result, metric_list
            )
            if all_match:
                matches.append(Match(doc_id, prompt, ref_result, target_result))
            else:
                deviations.append(
                    Deviation(doc_id, prompt, ref_result, target_result, worse_metrics)
                )

        return matches, deviations

    def _process_target_only_samples(
        self, target_data: Dict
    ) -> Tuple[List[Match], List[Deviation]]:
        """Process target samples against correct answers."""
        matches = []
        deviations = []

        for doc_id, target_result in target_data.items():
            prompt = self.get_prompt_text(target_result["doc"])

            # Use per-sample metric values stored by the harness (e.g. exact_match: 1.0)
            # rather than re-doing string comparison, so results align with harness scores.
            worse_metrics = [
                m["metric"]
                for m in (
                    self.metric_list
                    or [{"metric": "exact_match", "higher_is_better": True}]
                )
                if target_result.get(m["metric"], 1.0) == 0.0
            ]

            if not worse_metrics:
                matches.append(Match(doc_id, prompt, None, target_result))
            else:
                deviations.append(
                    Deviation(doc_id, prompt, None, target_result, worse_metrics)
                )

        return matches, deviations

    def _load_task_metrics(
        self, ref_dir: Path, target_dir: Path, task_name: str
    ) -> Dict:
        """Load task-specific metrics."""
        task_ref_metrics = {}
        task_target_metrics = {}

        if ref_dir:
            try:
                ref_results_path = self.get_results_file(ref_dir, task_name)
                task_ref_metrics = self.load_summary_metrics(ref_results_path)
                task_ref_metrics = {
                    k: v for k, v in task_ref_metrics.items() if k.startswith(task_name)
                }
            except FileNotFoundError as e:
                logger.warning(
                    "Reference results file not found for task %s: %s", task_name, e
                )
                pass

        try:
            target_results_path = self.get_results_file(target_dir, task_name)
            task_target_metrics = self.load_summary_metrics(target_results_path)
            task_target_metrics = {
                k: v for k, v in task_target_metrics.items() if k.startswith(task_name)
            }
        except FileNotFoundError as e:
            logger.warning(
                "Target results file not found for task %s: %s", task_name, e
            )
            pass

        return {"ref_metrics": task_ref_metrics, "target_metrics": task_target_metrics}

    def is_worse(self, ref_val, target_val, higher_is_better: bool) -> bool:
        """Determine if the target result is worse than the reference result."""
        return (target_val < ref_val) if higher_is_better else (target_val > ref_val)

    def compare_results(
        self, ref_result: Dict, target_result: Dict, metric_list: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """Compare metric values between reference and target results."""
        worse_metrics = []
        all_match = True
        for metric in metric_list:
            name = metric["metric"]
            ref_val = ref_result.get(name)
            target_val = target_result.get(name)
            if ref_val != target_val:
                all_match = False
                if self.is_worse(ref_val, target_val, metric["higher_is_better"]):
                    worse_metrics.append(name)
        return all_match, worse_metrics

    def analyze_all_results(
        self, ref_dir: Path, target_dir: Path
    ) -> Tuple[
        Dict[str, Tuple[List[Tuple[int, str, Dict, Dict]], List[Deviation], Dict]],
        List[Dict],
    ]:
        """Perform full comparison of reference vs target results across all matching task files."""
        # Handle case where ref_dir doesn't exist or is None
        if not ref_dir or not ref_dir.exists():
            return self._analyze_target_only(target_dir)

        available_tasks = self.get_available_tasks(ref_dir, target_dir)
        self.available_tasks = available_tasks

        if not available_tasks:
            return {}, []

        # Load metric list from the first available task
        metric_list = []
        for task_name in available_tasks:
            try:
                ref_results_path = self.get_results_file(ref_dir)
                metric_list = self.load_metric_list(ref_results_path, task_name)
                if metric_list:
                    break
            except FileNotFoundError as e:
                logger.warning(
                    "Reference results file not found in analyze_all_results: %s", e
                )
                continue

        if not metric_list:
            return {}, []

        matching_pairs = self.get_matching_sample_filepaths(ref_dir, target_dir)
        task_deviation_data = {}

        for ref_path, target_path in matching_pairs:
            task_name = self.get_task_name_from_path(ref_path)
            ref_data = self.load_samples_data(ref_path)
            target_data = self.load_samples_data(target_path)

            matches, deviations = self._process_task_samples(
                ref_data, target_data, metric_list
            )
            task_metrics = self._load_task_metrics(ref_dir, target_dir, task_name)
            task_deviation_data[task_name] = (matches, deviations, task_metrics)

        self.metric_list = metric_list
        return task_deviation_data, metric_list

    def _analyze_target_only(
        self, target_dir: Path
    ) -> Tuple[
        Dict[str, Tuple[List[Tuple[int, str, Dict, Dict]], List[Deviation], Dict]],
        List[Dict],
    ]:
        """Analyze target results against correct answers when no reference data is available."""
        if not target_dir or not target_dir.exists():
            return {}, []

        target_samples = list(target_dir.rglob("samples_*.jsonl"))
        if not target_samples:
            return {}, []

        metric_list = self._load_metric_list(target_dir)
        task_deviation_data = {}
        available_tasks = []

        for target_path in target_samples:
            task_name = self.get_task_name_from_path(target_path)
            available_tasks.append(task_name)

            target_data = self.load_samples_data(target_path)
            matches, deviations = self._process_target_only_samples(target_data)
            task_metrics = self._load_task_metrics(None, target_dir, task_name)
            task_deviation_data[task_name] = (matches, deviations, task_metrics)

        self.available_tasks = available_tasks
        self.metric_list = metric_list
        return task_deviation_data, metric_list

    def _create_entry(
        self,
        doc_id: int,
        ref_output: Dict,
        target_output: Dict,
        worse_metrics: List[str] = None,
    ) -> Dict:
        """Create a standardized entry for matching or deviating results."""
        # Use target_output as fallback when ref_output is None (target-only analysis)
        source_output = ref_output if ref_output is not None else target_output

        entry = {
            "doc_id": doc_id,
            "doc": source_output.get("doc", {}),
            "correct_answer": source_output.get("target", ""),
            "arguments": source_output.get("arguments", {}),
            "target": {
                "resps": target_output.get("resps", []),
                "filtered_resps": target_output.get("filtered_resps", []),
            },
            "filter": source_output.get("filter", ""),
            "metrics": source_output.get("metrics", []),
            "doc_hash": source_output.get("doc_hash", ""),
            "prompt_hash": source_output.get("prompt_hash", ""),
            "target_hash": source_output.get("target_hash", ""),
            "exact_match": target_output.get("exact_match", 0.0),
        }

        # Add reference section only if we have separate reference data
        if ref_output is not None and ref_output is not target_output:
            entry["reference"] = {
                "resps": ref_output.get("resps", []),
                "filtered_resps": ref_output.get("filtered_resps", []),
            }

        # Add worse_metrics for deviating entries
        if worse_metrics:
            entry["worse_metrics"] = worse_metrics

        return entry

    def save_eval_results_to_file(
        self,
        task_deviation_data: Dict[str, Tuple[List[Match], List[Deviation], Dict]],
        out_path: Path,
    ):
        """Write the comparison results to a JSON file."""
        results_json = {}

        for task_name, (
            matches,
            deviations,
            task_metrics,
        ) in task_deviation_data.items():
            results_json[task_name] = {
                "summary_metrics": {
                    "total_comparisons": len(matches) + len(deviations),
                    "matches": len(matches),
                    "deviations_with_worse_accuracy": sum(
                        1 for d in deviations if d.worse_metrics
                    ),
                },
                "ref_metrics": task_metrics.get("ref_metrics", {}),
                "target_metrics": task_metrics.get("target_metrics", {}),
                "matching": {},
                "deviating": {},
            }

            # Add matching entries
            for match in matches:
                if (
                    match.ref_results is None or isinstance(match.ref_results, dict)
                ) and isinstance(match.target_results, dict):
                    entry = self._create_entry(
                        match.doc_id, match.ref_results, match.target_results
                    )
                else:
                    entry = {"doc_id": match.doc_id, "prompt": match.prompt}
                results_json[task_name]["matching"][str(match.doc_id)] = entry

            # Add deviating entries
            for d in deviations:
                if (
                    d.worse_metrics
                    and (d.ref_results is None or isinstance(d.ref_results, dict))
                    and isinstance(d.target_results, dict)
                ):
                    entry = self._create_entry(
                        d.doc_id, d.ref_results, d.target_results, d.worse_metrics
                    )
                else:
                    entry = {
                        "doc_id": d.doc_id,
                        "prompt": d.prompt,
                        "worse_metrics": d.worse_metrics,
                    }
                results_json[task_name]["deviating"][str(d.doc_id)] = entry

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)

    def create_aggregate_entry(self, prefix, aggregate_name, task_deviation_data):
        """Create aggregated entry for tasks with given prefix"""
        tasks = [task for task in task_deviation_data.keys() if task.startswith(prefix)]
        if not tasks:
            return None

        all_matches, all_deviations = [], []
        target_scores, ref_scores = [], []

        for task in tasks:
            matches, deviations, task_metrics = task_deviation_data[task]
            all_matches.extend(matches)
            all_deviations.extend(deviations)

            # Get first non-stderr metric as target score
            for value in task_metrics.get("target_metrics", {}).values():
                if isinstance(value, (int, float)):
                    target_scores.append(value)
                    break

            # Get first non-stderr metric as ref score
            for value in task_metrics.get("ref_metrics", {}).values():
                if isinstance(value, (int, float)):
                    ref_scores.append(value)
                    break

        avg_target_score = (
            sum(target_scores) / len(target_scores) if target_scores else 0
        )
        avg_ref_score = sum(ref_scores) / len(ref_scores) if ref_scores else None

        combined_metrics = {
            "target_metrics": {f"{aggregate_name}-avg_score": avg_target_score},
            "ref_metrics": {f"{aggregate_name}-avg_score": avg_ref_score}
            if avg_ref_score is not None
            else {},
        }

        return (all_matches, all_deviations, combined_metrics)

    def apply_aggregation(self, task_deviation_data):
        """Apply aggregation to task data"""
        aggregated_task_data = {}

        # Add non-subdataset tasks
        aggregatable_prefixes = ["mmlu_pro_", "bbh_cot_fewshot_"]
        for task_name, data in task_deviation_data.items():
            if not any(
                task_name.startswith(prefix) for prefix in aggregatable_prefixes
            ):
                aggregated_task_data[task_name] = data

        # Create aggregated entries
        aggregates = [
            ("mmlu_pro_", "mmlu_pro", "mmlu_pro_aggregate"),
            ("bbh_cot_fewshot_", "bbh_cot_fewshot", "bbh_cot_fewshot_aggregate"),
        ]

        for prefix, name, aggregate_key in aggregates:
            entry = self.create_aggregate_entry(prefix, name, task_deviation_data)
            if entry:
                aggregated_task_data[aggregate_key] = entry

        return aggregated_task_data

    def create_results_json(self, task_deviation_data):
        """Create results JSON for summary display"""
        results_json = {}
        for task_name, (
            matches,
            deviations,
            task_metrics,
        ) in task_deviation_data.items():
            results_json[task_name] = {
                "summary_metrics": {
                    "total_comparisons": len(matches) + len(deviations),
                    "matches": len(matches),
                    "deviations_with_worse_accuracy": sum(
                        1 for d in deviations if d.worse_metrics
                    ),
                },
                "ref_metrics": task_metrics.get("ref_metrics", {}),
                "target_metrics": task_metrics.get("target_metrics", {}),
            }
        return results_json

    def print_summary(self, results_json, aggregate=False):
        """Print analysis summary and metrics table"""
        print("\n=== TASK ANALYZER RESULTS SUMMARY ===")
        print(f"Total tasks analyzed: {len(results_json)}")

        # Summary by task category
        categories = {"GSM8K": [], "BBH": [], "MMLU Pro": [], "Other": []}
        for task in results_json.keys():
            if "gsm8k" in task:
                categories["GSM8K"].append(task)
            elif "bbh_cot_fewshot" in task:
                categories["BBH"].append(task)
            elif "mmlu_pro" in task:
                categories["MMLU Pro"].append(task)
            else:
                categories["Other"].append(task)

        for category, tasks in categories.items():
            if tasks:
                print(f"{category}: {len(tasks)} tasks")

        self._print_metrics_table(results_json, aggregate)

    def _print_metrics_table(self, results_json, aggregate):
        """Print the metrics comparison table"""
        print("\n=== KEY PERFORMANCE METRICS ===")

        key_tasks = self._get_key_tasks(results_json, aggregate)
        has_ref_data = any(
            results_json[task].get("ref_metrics")
            for task in key_tasks
            if task in results_json
        )

        # Print table header
        if has_ref_data:
            print(
                f"\n{'Dataset':<30} {'Primary Metric':<25} {'Ref Score':<10} {'Target Score':<12} {'Samples':<8} {'Matches':<8} {'Accuracy':<10}"
            )
            print("-" * 103)
        else:
            print(
                f"\n{'Dataset':<30} {'Primary Metric':<25} {'Score':<10} {'Samples':<8} {'Correct':<8} {'Accuracy':<10}"
            )
            print("-" * 87)

        # Print table rows
        for task in key_tasks:
            if task in results_json:
                self._print_task_row(task, results_json[task], has_ref_data)

        # Print detailed metrics
        self._print_detailed_metrics(key_tasks, results_json)

    def _get_key_tasks(self, results_json, aggregate):
        """Get list of key tasks to display based on aggregation setting"""
        default_categories = [
            "bbh_cot_fewshot",
            "gpqa_main_cot_n_shot",
            "gsm8k_cot",
            "leaderboard_ifeval",
            "mbpp",
            "mmlu_pro",
        ]
        aggregatable_categories = {
            "mmlu_pro": ("mmlu_pro_", "mmlu_pro_aggregate"),
            "bbh_cot_fewshot": ("bbh_cot_fewshot_", "bbh_cot_fewshot_aggregate"),
        }

        key_tasks = []
        for category in default_categories:
            if aggregate and category in aggregatable_categories:
                prefix, aggregate_key = aggregatable_categories[category]
                entry = self._create_aggregate_display_entry(
                    prefix, category, results_json
                )
                if entry:
                    results_json[aggregate_key] = entry
                    key_tasks.append(aggregate_key)
            else:
                if category in aggregatable_categories and not aggregate:
                    prefix, _ = aggregatable_categories[category]
                    tasks = [
                        task
                        for task in results_json.keys()
                        if task.startswith(prefix)
                        and results_json[task].get("target_metrics")
                    ]
                    key_tasks.extend(tasks)
                else:
                    for task in results_json.keys():
                        if category in task and results_json[task].get(
                            "target_metrics"
                        ):
                            key_tasks.append(task)
                            break
        return key_tasks

    def _create_aggregate_display_entry(self, prefix, aggregate_name, results_json):
        """Create aggregated display entry for tasks with given prefix"""
        tasks = [
            task
            for task in results_json.keys()
            if task.startswith(prefix) and results_json[task].get("target_metrics")
        ]
        if not tasks:
            return None

        total_samples = sum(
            results_json[task]["summary_metrics"]["total_comparisons"] for task in tasks
        )
        total_matches = sum(
            results_json[task]["summary_metrics"]["matches"] for task in tasks
        )

        target_scores, ref_scores = [], []
        for task in tasks:
            for value in results_json[task]["target_metrics"].values():
                if isinstance(value, (int, float)):
                    target_scores.append(value)
                    break
            for value in results_json[task].get("ref_metrics", {}).values():
                if isinstance(value, (int, float)):
                    ref_scores.append(value)
                    break

        avg_target_score = (
            sum(target_scores) / len(target_scores) if target_scores else 0
        )
        avg_ref_score = sum(ref_scores) / len(ref_scores) if ref_scores else None

        return {
            "summary_metrics": {
                "total_comparisons": total_samples,
                "matches": total_matches,
            },
            "target_metrics": {f"{aggregate_name}-avg_score": avg_target_score},
            "ref_metrics": {f"{aggregate_name}-avg_score": avg_ref_score}
            if avg_ref_score is not None
            else {},
        }

    def _print_task_row(self, task, task_data, has_ref_data):
        """Print a single task row in the metrics table"""
        metrics = task_data.get("target_metrics", {})
        ref_metrics = task_data.get("ref_metrics", {})
        summary = task_data.get("summary_metrics", {})

        # Get primary metric
        primary_metric = None
        target_score = None
        ref_score = None
        for metric_name, value in metrics.items():
            if "stderr" not in metric_name:
                primary_metric = (
                    metric_name.split("-")[1] if "-" in metric_name else metric_name
                )
                target_score = value
                ref_score = ref_metrics.get(metric_name, None)
                break

        total = summary.get("total_comparisons", 0)
        matches = summary.get("matches", 0)
        accuracy = f"{matches / total * 100:.1f}%" if total > 0 else "N/A"
        target_score_str = f"{target_score:.3f}" if target_score else "N/A"
        ref_score_str = f"{ref_score:.3f}" if ref_score else "N/A"

        if has_ref_data:
            print(
                f"{task:<30} {primary_metric or 'N/A':<25} {ref_score_str:<10} {target_score_str:<12} {total:<8} {matches:<8} {accuracy:<10}"
            )
        else:
            print(
                f"{task:<30} {primary_metric or 'N/A':<25} {target_score_str:<10} {total:<8} {matches:<8} {accuracy:<10}"
            )

    def _print_detailed_metrics(self, key_tasks, results_json):
        """Print detailed metrics section"""
        print("\n=== DETAILED METRICS ===")
        for task in key_tasks:
            if task in results_json:
                metrics = results_json[task].get("target_metrics", {})
                print(f"\n{task.upper()}:")
                for metric_name, value in metrics.items():
                    if "stderr" not in metric_name:
                        print(f"  {metric_name}: {value:.3f}")

    def run_analysis(self, ref_path, target_path, output_path, aggregate=False):
        """Run complete analysis workflow"""
        # Analyze results
        task_deviation_data, _ = self.analyze_all_results(ref_path, target_path)

        # Apply aggregation if requested
        if aggregate:
            task_deviation_data = self.apply_aggregation(task_deviation_data)

        # Save results
        self.save_eval_results_to_file(task_deviation_data, output_path)

        # Create and print summary
        results_json = self.create_results_json(task_deviation_data)
        self.print_summary(results_json, aggregate)

        print(f"\nResults saved to {output_path}")
        return results_json


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(description="Run task evaluation and analysis.")
    parser.add_argument(
        "--ref-result-path", help="Path to reference evaluation results"
    )
    parser.add_argument(
        "--target-results-path", required=True, help="Path to target evaluation results"
    )
    parser.add_argument(
        "--output-path", default="./comparison_results.json", help="Output file path"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate subdatasets into meta datasets",
    )
    args = parser.parse_args()

    analyzer = LmEvalAnalyzer(
        ref_result_path=args.ref_result_path,
        target_results_path=args.target_results_path,
    )

    analyzer.run_analysis(
        ref_path=Path(args.ref_result_path) if args.ref_result_path else None,
        target_path=Path(args.target_results_path),
        output_path=Path(args.output_path),
        aggregate=args.aggregate,
    )


if __name__ == "__main__":
    main()
