import sys
from pathlib import Path

import pytest

from lift_bench.hierarchical_drafting.pipeline import cli
from lift_bench.hierarchical_drafting.utils.file_utils import load_yaml

CONFIGS = sorted((Path(__file__).parents[3] / "configs" / "published_splits").glob("*.yaml"))


def test_published_split_configs_exist():
    assert len(CONFIGS) == 10


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_parsing_reproduces_every_configured_value(config_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--config", str(config_path)])

    params = cli.parse_args().model_dump()

    config = load_yaml(str(config_path))
    assert {group: {k: params[group][k] for k in fields} for group, fields in config.items()} == config


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.stem)
def test_task_shape_is_pinned_rather_than_left_to_defaults(config_path):
    drafting_task = load_yaml(str(config_path))["drafting_task"]

    assert {"task_mode", "task_verbalizer_name", "content_weight"} <= set(drafting_task)
