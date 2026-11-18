# SPDX-License-Identifier: Apache-2.0
"""Run the verifier on one short case, dumping rank 0's per-layer streams.

    DEEPSEEK_V41_MEGAKERNEL=1 DEEPSEEK_V41_DUMP_DIR=/mnt/nvme/dump_mk \\
        python verify_short.py --model /mnt/nvme/hf [--case trivia-capital-france]

Then `python compare_dumps.py /mnt/nvme/dump_op /mnt/nvme/dump_mk` prints the
per-layer agreement of the two paths on the same prompt.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_TEST_ROOT = Path(__file__).resolve().parent.parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

import model_package  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--case", default="trivia-capital-france")
    ap.add_argument("--report", default=None)
    ap.add_argument("--max-model-len", type=int, default=608)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    args = ap.parse_args()
    model_package.load()
    from deepseek_v41.verifier import prompts, run_verifier as rv
    from deepseek_v41.verifier.run_verifier import main as run_verifier

    wanted = set(args.case.split(","))
    # The jsonl cases are picked by SELECTED_IDS; the two inline ones (smoke-test,
    # trivia-capital-france) are always built, so filter the built set instead.
    prompts.SELECTED_IDS = tuple(c for c in wanted if c not in ("smoke-test", "trivia-capital-france"))
    orig = prompts.build_cases

    def filtered():
        cs = orig()
        cs.cases = [c for c in cs.cases if c.id in wanted]
        return cs

    prompts.build_cases = filtered
    if hasattr(rv, "build_cases"):
        rv.build_cases = filtered
    work = tempfile.mkdtemp(prefix="verify_short_")
    os.chdir(work)
    report = args.report or os.path.join(work, "verifier_report.json")
    rc = run_verifier(["--model", args.model, "--report", report, "--max-model-len", str(args.max_model_len),
                       "--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    print(f"verify_short: rc={rc} report={report}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
