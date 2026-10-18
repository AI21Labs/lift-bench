"""Evaluate a few tasks of every LIFT-Bench config and split, loaded from the Hugging Face Hub with `datasets`.

Exercises the `load_dataset` path end to end: each (config, split) is loaded from the Hub, the first N rows are
written back to JSONL as `datasets` returns them, and the `drafting_constraint` taskset scores them through
`vf-eval`, exactly as `run_eval.sh` does. Runs locally (null harness, subprocess runtime, no push).

    export OPENAI_BASE_URL=... OPENAI_API_KEY=...
    .venv/bin/python scripts/evaluate_hub_splits.py [--per-split 5] [--model <name>] [--judge-model <name>]

Results land in --output-dir (default .results/hub_splits/<config>/<split>/), one vf-eval run per (config, split).
The exit code is 1 when any task ended unscored.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "ai21labs/LIFTBench"
DEFAULT_MODEL = "vllm-thinkingmachines-inkling-small"
SPLITS = ("train", "test")


@dataclass(frozen=True)
class RunSummary:
    config: str
    split: str
    rewards: list[float]
    failed: int
    checks: int
    errors: set[str]

    @property
    def tasks(self) -> int:
        return len(self.rewards) + self.failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hub dataset repository")
    parser.add_argument("--configs", nargs="*", help="configs to evaluate (default: every config of the repo)")
    parser.add_argument("--splits", nargs="*", default=SPLITS, choices=SPLITS)
    parser.add_argument("--per-split", type=int, default=5, help="tasks to evaluate from each split")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model under test")
    parser.add_argument("--judge-model", default=None, help="judge override (default: the model under test)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".results" / "hub_splits")
    args = parser.parse_args()
    args.judge_model = args.judge_model or args.model
    return args


def write_split(repo: str, config: str, split: str, per_split: int, path: Path) -> int:
    rows = load_dataset(repo, config, split=f"{split}[:{per_split}]")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def run_vf_eval(dataset_path: Path, run_dir: Path, args: argparse.Namespace) -> None:
    base_url = os.environ.get("VLLM_BASE_URL") or os.environ["OPENAI_BASE_URL"]
    api_key_var = "VLLM_API_KEY" if os.environ.get("VLLM_API_KEY") else "OPENAI_API_KEY"
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "vf-eval"),
        "drafting_constraint",
        *("--model", args.model),
        *("--client.base-url", base_url),
        *("--client.api-key-var", api_key_var),
        *("--env.agent.harness.id", "null"),
        *("--env.agent.runtime.type", "subprocess"),
        "--no-push",
        *("--select.limit", str(args.per_split)),
        *("--num-rollouts", "1"),
        *("--output-dir", str(run_dir)),
        "--no-rich",
        *("--env.taskset.dataset-path", str(dataset_path)),
        *("--env.taskset.judge-model", args.judge_model),
    ]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "environments")])}
    with (run_dir / "vf-eval.log").open("w") as log:
        subprocess.run(command, env=env, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def summarize(run_dir: Path, config: str, split: str) -> RunSummary:
    # A rerun into the same output dir adds a run; report the newest.
    traces_path = max(run_dir.glob("drafting_constraint--*/traces.jsonl"), key=lambda p: p.stat().st_mtime)
    rewards: list[float] = []
    failed = checks = 0
    errors: set[str] = set()
    with traces_path.open() as f:
        for line in f:
            episode = json.loads(line)
            (trace,) = episode["traces"]
            checks += len(trace["info"].get("agg_eval_scores") or [])
            if episode["ok"]:
                rewards.append(trace["rewards"]["constraint_reward"]["score"])
            else:
                failed += 1
                errors.update(error.get("type", "?") for error in episode.get("errors", []))
    return RunSummary(config, split, rewards, failed, checks, errors)


def print_report(summaries: list[RunSummary]) -> None:
    print(f"\n{'config':32s} {'split':5s} tasks  ok  mean   min    max   checks")
    for s in summaries:
        stats = (
            f"{statistics.fmean(s.rewards):.3f}  {min(s.rewards):.3f}  {max(s.rewards):.3f}"
            if s.rewards
            else f"{'-':>5s}  {'-':>5s}  {'-':>5s}"
        )
        errors = f"  errors={','.join(sorted(s.errors))}" if s.errors else ""
        print(f"{s.config:32s} {s.split:5s} {s.tasks:5d}  {s.tasks - s.failed:2d}  {stats}  {s.checks:5d}{errors}")
    rewards = [r for s in summaries for r in s.rewards]
    mean = f"mean reward {statistics.fmean(rewards):.3f}" if rewards else "no task was scored"
    print(
        f"\n{len(summaries)} runs, {sum(s.tasks for s in summaries)} tasks, {sum(s.failed for s in summaries)} "
        f"unscored, {sum(s.checks for s in summaries)} checks, {mean}"
    )


def main() -> int:
    args = parse_args()
    configs = args.configs or get_dataset_config_names(args.repo)
    summaries = []
    for config in configs:
        for split in args.splits:
            run_dir = args.output_dir / config / split
            run_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = run_dir / f"{split}.jsonl"
            n = write_split(args.repo, config, split, args.per_split, dataset_path)
            print(f"{config}/{split}: {n} tasks loaded from {args.repo}, evaluating...", flush=True)
            run_vf_eval(dataset_path, run_dir, args)
            summaries.append(summarize(run_dir, config, split))
    print_report(summaries)
    return 1 if any(s.failed for s in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())
