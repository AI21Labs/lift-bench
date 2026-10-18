from types import SimpleNamespace
from typing import Callable

import pytest
import verifiers.v1 as vf

from lift_bench.hierarchical_drafting.metrics import critic


@pytest.fixture
def judge(monkeypatch):
    """Fake judge model: replies with the queued `decisions` and records every (model, prompt) it was sent."""
    state = SimpleNamespace(decisions=["<decision>1</decision>"], prompts=[], fail=False)

    async def fake_generate_completions(model, prompts, n=1, **_):
        if state.fail:
            raise RuntimeError("gateway unavailable")
        state.prompts.append((model, prompts[0][0]["content"]))
        return [SimpleNamespace(completions=state.decisions[:n])]

    monkeypatch.setattr(critic, "generate_completions", fake_generate_completions)
    return state


@pytest.fixture
def trace_with_reply() -> Callable[[vf.Task, str], vf.Trace]:
    """A rollout trace of `task` answered by a hardcoded reply, to score without model inference."""

    def build(task: vf.Task, reply: str) -> vf.Trace:
        return vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
            nodes=[
                vf.MessageNode(parent=None, message=vf.UserMessage(content="prompt")),
                vf.MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
            ],
        )

    return build
