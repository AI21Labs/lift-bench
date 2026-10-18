import sys

import pytest
from pydantic import ValidationError

from lift_bench.hierarchical_drafting.iftrain.instructions_registry import INSTRUCTION_DICT
from lift_bench.hierarchical_drafting.pipeline import cli


def _parse(monkeypatch, *argv: str) -> cli.Params:
    monkeypatch.setattr(sys, "argv", ["main", *argv])
    return cli.parse_args()


def _config(tmp_path, text: str, suffix: str = ".yaml") -> str:
    path = tmp_path / f"config{suffix}"
    path.write_text(text)
    return str(path)


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_config_file_values_override_defaults(tmp_path, monkeypatch, suffix):
    config = _config(
        tmp_path, "drafting_task:\n  naturalize_constraints: true\npipeline_infra:\n  max_examples: 7\n", suffix
    )

    params = _parse(monkeypatch, "--config", config)

    assert params.drafting_task.naturalize_constraints
    assert params.pipeline_infra.max_examples == 7


def test_explicit_cli_args_override_the_config_file(tmp_path, monkeypatch):
    config = _config(tmp_path, "pipeline_infra:\n  max_examples: 7\n  pipeline_name: from_config\n")

    params = _parse(monkeypatch, "--config", config, "--pipeline_infra.max_examples", "9")

    assert params.pipeline_infra.max_examples == 9
    assert params.pipeline_infra.pipeline_name == "from_config"


@pytest.mark.parametrize("name", ["config.json", "config.toml"])
def test_non_yaml_config_files_are_rejected(tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_text("{}")

    with pytest.raises(ValueError, match="Unknown config file type"):
        _parse(monkeypatch, "--config", str(path))


@pytest.mark.parametrize(
    "text",
    [
        "ifbench_constraint_sampling:\n  ifbench_min_complexity: 0\n",
        "max_examples: 7\n",
        "no_such_group:\n  max_examples: 7\n",
    ],
    ids=["unknown-key-in-group", "key-outside-its-group", "unknown-group"],
)
def test_unknown_config_keys_are_rejected(tmp_path, monkeypatch, text):
    with pytest.raises(ValidationError):
        _parse(monkeypatch, "--config", _config(tmp_path, text))


def test_list_args_accept_comma_separated_values(monkeypatch):
    params = _parse(
        monkeypatch,
        "--ifbench_constraint_sampling.use_ifbench_constraints",
        "--ifbench_constraint_sampling.ifbench_custom_instructions_lst",
        "copy:repeat_phrase,keywords:palindrome",
    )

    assert params.ifbench_constraint_sampling.use_ifbench_constraints
    assert params.ifbench_constraint_sampling.ifbench_custom_instructions_lst == [
        "copy:repeat_phrase",
        "keywords:palindrome",
    ]


def test_ifbench_constraints_are_off_by_default(monkeypatch):
    ifbench = _parse(monkeypatch).ifbench_constraint_sampling

    assert not ifbench.use_ifbench_constraints
    assert len(ifbench.ifbench_custom_instructions_lst) == 19
    assert set(ifbench.ifbench_custom_instructions_lst) <= INSTRUCTION_DICT.keys()


@pytest.mark.parametrize("group", ["content_eval_type", "artifact_eval_type"])
def test_only_implemented_critics_are_accepted(group, monkeypatch):
    with pytest.raises(ValidationError):
        _parse(monkeypatch, f"--llm_judge_constraint_generation.{group}", "absolute_critic")


@pytest.mark.parametrize(("value", "expected"), [("100", 100), ("0.1", 0.1)])
def test_split_sizes_keep_counts_as_ints_and_fractions_as_floats(monkeypatch, value, expected):
    test_size = _parse(monkeypatch, "--pipeline_infra.test_size", value).pipeline_infra.test_size

    assert test_size == expected
    assert type(test_size) is type(expected)


def test_flags_that_default_to_true_can_be_turned_off(monkeypatch):
    verifiable = _parse(
        monkeypatch,
        "--no-verifiable_constraint_sampling.use_strict_header_names",
        "--no-verifiable_constraint_sampling.verified_exact_sampling",
    ).verifiable_constraint_sampling

    assert not verifiable.use_strict_header_names
    assert not verifiable.verified_exact_sampling


def test_flat_mode_drops_the_outline_and_uses_a_flat_verbalizer(monkeypatch):
    params = _parse(monkeypatch, "--drafting_task.task_mode", "flat")

    assert (params.doc_outline.doc_template_max_children, params.doc_outline.doc_template_max_depth) == (0, 0)
    assert not params.verifiable_constraint_sampling.use_strict_header_names
    assert params.drafting_task.task_verbalizer_name == "ifeval_flat"


def test_flat_mode_keeps_an_explicitly_chosen_flat_verbalizer(monkeypatch):
    params = _parse(monkeypatch, "--drafting_task.task_mode", "flat", "--drafting_task.task_verbalizer_name", "flat")

    assert params.drafting_task.task_verbalizer_name == "flat"
