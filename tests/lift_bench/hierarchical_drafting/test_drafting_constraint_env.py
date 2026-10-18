import json
import re

import drafting_constraint as drafting_constraint_module
import pytest
import verifiers.v1 as vf

PROMPT = [{"role": "user", "content": "Write a greeting."}]
EVALUATIONS = [
    {
        "metric_name": "emoji_count",
        "metric_args": {"lower_bound": 1, "upper_bound": 3},
        "weight": 0.5,
        "evaluation_id": "content_emoji_count_0",
    },
    {
        "metric_name": "lexical_match",
        "metric_args": {"string": "hello", "match_type": "contains", "should_match": True, "fuzzy_match": True},
        "weight": 0.5,
        "evaluation_id": "content_lexical_match_1",
    },
]
TASKS = [
    {"example_id": "a1", "topic": "Greetings", "prompt": PROMPT, "info": {"evaluations": EVALUATIONS}},
    {"example_id": "b2", "topic": "Farewells", "prompt": PROMPT, "info": {"evaluations": EVALUATIONS[1:]}},
]


@pytest.fixture
def drafting_constraint():
    return drafting_constraint_module


@pytest.fixture
def split_path(tmp_path) -> str:
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(task) + "\n" for task in TASKS))
    return str(path)


def _tasks(drafting_constraint, split_path, **config) -> list:
    config = drafting_constraint.DraftingConstraintConfig(dataset_path=split_path, **config)
    return list(drafting_constraint.DraftingConstraintTaskset(config))


def test_every_task_keeps_exactly_its_own_evaluations(drafting_constraint, split_path):
    tasks = _tasks(drafting_constraint, split_path)

    assert [task.data.evaluations for task in tasks] == [task["info"]["evaluations"] for task in TASKS]


def test_a_missing_dataset_path_is_reported(drafting_constraint):
    with pytest.raises(ValueError, match=re.escape("--env.taskset.dataset-path")):
        list(drafting_constraint.DraftingConstraintTaskset(drafting_constraint.DraftingConstraintConfig()))


@pytest.fixture
def hub_downloads(monkeypatch, split_path) -> list[dict]:
    """Fake `hf_hub_download`: records each call and serves the local split file."""
    calls: list[dict] = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        return split_path

    monkeypatch.setattr(drafting_constraint_module, "hf_hub_download", fake_hf_hub_download)
    return calls


def test_a_hub_split_is_downloaded_at_the_pinned_revision_and_loaded(drafting_constraint, split_path, hub_downloads):
    config = drafting_constraint.DraftingConstraintConfig(
        dataset_config="short_doc_some_constraints", dataset_split="train", dataset_revision="abc123"
    )

    tasks = list(drafting_constraint.DraftingConstraintTaskset(config))

    assert hub_downloads == [
        {
            "repo_id": "ai21labs/LIFTBench",
            "filename": "short_doc_some_constraints/train.jsonl",
            "repo_type": "dataset",
            "revision": "abc123",
        }
    ]
    assert [task.data for task in tasks] == [task.data for task in _tasks(drafting_constraint, split_path)]


def test_a_hub_split_without_a_revision_reads_the_default_branch(drafting_constraint, hub_downloads):
    config = drafting_constraint.DraftingConstraintConfig(
        dataset_config="short_doc_some_constraints", dataset_split="test"
    )

    list(drafting_constraint.DraftingConstraintTaskset(config))

    assert hub_downloads[0]["revision"] is None


@pytest.mark.parametrize(
    "config",
    [
        {"dataset_path": "test.jsonl", "dataset_config": "short_doc_some_constraints", "dataset_split": "test"},
        {"dataset_config": "short_doc_some_constraints"},
    ],
    ids=["both-sources", "config-without-split"],
)
def test_an_ambiguous_or_incomplete_split_source_is_reported(drafting_constraint, hub_downloads, config):
    with pytest.raises(ValueError, match=re.escape("--env.taskset.dataset-")):
        list(drafting_constraint.DraftingConstraintTaskset(drafting_constraint.DraftingConstraintConfig(**config)))

    assert hub_downloads == []


def test_tasks_are_sent_as_generated(drafting_constraint, split_path):
    (task, _) = _tasks(drafting_constraint, split_path)

    assert [m.model_dump(exclude_none=True) for m in task.data.prompt] == PROMPT
    assert task.data.system_prompt is None


def test_tasks_are_keyed_by_example_id(drafting_constraint, split_path):
    assert [task.key for task in _tasks(drafting_constraint, split_path)] == ["a1", "b2"]


def test_a_system_prompt_file_is_set_on_every_task(drafting_constraint, split_path, tmp_path):
    system_prompt = tmp_path / "system.txt"
    system_prompt.write_text("Be precise.")

    tasks = _tasks(drafting_constraint, split_path, system_prompt=system_prompt)

    assert [task.data.system_prompt for task in tasks] == ["Be precise.", "Be precise."]


async def test_rollouts_are_scored_and_per_evaluation_scores_kept(drafting_constraint, split_path, trace_with_reply):
    (task, _) = _tasks(drafting_constraint, split_path)
    trace = trace_with_reply(task, "hello there 👋")

    await task.score(trace)

    assert trace.reward == 1.0
    assert trace.info["agg_eval_scores"] == [1.0, 1.0]
    assert trace.info["agg_eval_ids"] == ["content_emoji_count_0", "content_lexical_match_1"]


async def test_a_rollout_with_an_errored_check_is_flagged_scores_zero_and_keeps_scores(
    drafting_constraint, tmp_path, trace_with_reply
):
    broken_check = {**EVALUATIONS[0], "metric_args": {"lower_bound": 3, "upper_bound": 1}}
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps({**TASKS[0], "info": {"evaluations": [broken_check, EVALUATIONS[1]]}}) + "\n")
    (task,) = _tasks(drafting_constraint, str(path))
    trace = trace_with_reply(task, "hello there 👋")

    with pytest.raises(vf.TaskError, match="1 of 2 evaluations failed"):
        await task.score(trace)

    assert trace.reward == 0.0
    assert trace.info["agg_eval_scores"] == [None, 1.0]


JUDGED_TASK = {
    "example_id": "c3",
    "topic": "Greetings",
    "prompt": PROMPT,
    "info": {
        "evaluations": [
            {
                "metric_name": "light_absolute_critic",
                "metric_args": {"critic_model_name": "stored-judge", "criteria": ["Greets the reader."]},
                "weight": 1.0,
                "evaluation_id": "content_judge_0",
            }
        ]
    },
}


@pytest.fixture
def judged_split_path(tmp_path) -> str:
    path = tmp_path / "judged.jsonl"
    path.write_text(json.dumps(JUDGED_TASK) + "\n")
    return str(path)


async def test_the_judge_model_setting_scores_judge_checks_with_that_model(
    drafting_constraint, judged_split_path, trace_with_reply, judge
):
    (task,) = _tasks(drafting_constraint, judged_split_path, judge_model="configured-judge")

    await task.score(trace_with_reply(task, "hello there"))

    assert [model for model, _ in judge.prompts] == ["configured-judge"]
    assert task.data.judge_model == "configured-judge"


async def test_without_the_judge_model_setting_the_split_names_the_judge(
    drafting_constraint, judged_split_path, trace_with_reply, judge
):
    (task,) = _tasks(drafting_constraint, judged_split_path)

    await task.score(trace_with_reply(task, "hello there"))

    assert [model for model, _ in judge.prompts] == ["stored-judge"]
    assert task.data.judge_model is None
