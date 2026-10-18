"""End-to-end integration test: one vf-eval rollout of the dummy taskset against a live model.

Run only when an OpenAI-compatible server (e.g. vLLM) is available:

    pytest -m integration -v tests/lift_bench/hierarchical_drafting/test_constraint_reward_e2e.py

Configuration (override via env vars):
    OPENAI_BASE_URL  default: http://localhost:8000/v1 (a local vLLM server)
    OPENAI_API_KEY   the endpoint's API key
    VLLM_BASE_URL    endpoint of the model under test, when it differs from the judge's (default: OPENAI_BASE_URL)
    VLLM_API_KEY     its API key (default: OPENAI_API_KEY)
    VLLM_MODEL       default: Qwen/Qwen2.5-32B-Instruct

The model's answer is non-deterministic, so assertions are structural (reward in [0, 1], per-evaluation
scores kept) rather than exact. The run stays local: it isn't pushed and runs in a subprocess runtime.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.utils.trace_store import read_episodes

REPO_ROOT = Path(__file__).parents[3]
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
_VLLM_API_KEY_VAR = "VLLM_API_KEY" if os.environ.get("VLLM_API_KEY") else "OPENAI_API_KEY"


@pytest.fixture(scope="module")
def trace(tmp_path_factory) -> vf.Trace:
    output_dir = tmp_path_factory.mktemp("eval")
    command = [
        str(Path(sys.executable).parent / "vf-eval"),
        "drafting_constraint_dummy",
        *("-m", VLLM_MODEL),
        *("--client.base-url", VLLM_BASE_URL),
        *("--client.api-key-var", _VLLM_API_KEY_VAR),
        *("--env.agent.harness.id", "null"),
        *("--env.agent.runtime.type", "subprocess"),
        "--no-push",
        *("-n", "1", "-r", "1"),
        *("--sampling.temperature", "0"),
        *("-o", str(output_dir)),
        "--no-rich",
    ]
    env = {
        **os.environ,
        # vLLM accepts any non-empty API key.
        _VLLM_API_KEY_VAR: os.environ.get(_VLLM_API_KEY_VAR, "EMPTY"),
    }
    subprocess.run(command, env=env, cwd=REPO_ROOT, check=True)

    (run_dir,) = output_dir.iterdir()
    (episode,) = read_episodes(run_dir, vf.Trace[vf.WireTaskData, vf.State, vf.AgentConfig])
    (trace,) = episode.traces
    return trace


@pytest.mark.integration
def test_the_rollout_is_scored_without_error(trace: vf.Trace) -> None:
    assert trace.ok and not trace.errors
    assert 0.0 <= trace.reward <= 1.0, f"reward {trace.reward!r} is out of [0, 1]"


@pytest.mark.integration
def test_per_evaluation_scores_are_kept(trace: vf.Trace) -> None:
    # The dummy task's evaluations carry no evaluation_id, so there are no ids to keep.
    assert trace.info["agg_eval_ids"] is None
    assert len(trace.info["agg_eval_scores"]) == 3


@pytest.mark.integration
def test_the_task_prompt_is_sent_as_is(trace: vf.Trace) -> None:
    sent = [m.model_dump(exclude_none=True) for m in trace.messages[: len(trace.task.data.prompt)]]

    assert sent == [m.model_dump(exclude_none=True) for m in trace.task.data.prompt]


@pytest.mark.integration
def test_the_answer_is_non_empty(trace: vf.Trace) -> None:
    assert trace.assistant_messages and (trace.assistant_messages[-1].content or "").strip()
