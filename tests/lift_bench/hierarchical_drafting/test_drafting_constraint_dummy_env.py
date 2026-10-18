import drafting_constraint_dummy
import pytest
import verifiers.v1 as vf


@pytest.fixture
def dummy_env():
    return drafting_constraint_dummy


def test_the_task_keeps_exactly_its_own_evaluations(dummy_env):
    (task,) = dummy_env.DraftingConstraintDummyTaskset(vf.TasksetConfig())

    assert task.data.evaluations == dummy_env._ANSWER["evaluations"]


async def test_an_answer_meeting_every_constraint_scores_one(dummy_env, trace_with_reply):
    (task,) = dummy_env.DraftingConstraintDummyTaskset(vf.TasksetConfig())
    trace = trace_with_reply(task, "hello there 👋")

    await task.score(trace)

    assert not trace.errors
    assert trace.reward == pytest.approx(1.0)
