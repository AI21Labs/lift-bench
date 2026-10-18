import random
from typing import Type

from lift_bench.hierarchical_drafting.pipeline.task_verbalization.base import (
    TaskVerbalizer,
)

_rng = random.Random(1337)

_REGISTRY: dict[str, Type[TaskVerbalizer]] = {}


def register_task_verbalizer(name: str):
    def decorator(cls: Type[TaskVerbalizer]):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_task_verbalizer(name: str) -> TaskVerbalizer:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task verbalizer: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def sample_task_verbalizer() -> TaskVerbalizer:
    name = _rng.choice(list(_REGISTRY))
    return _REGISTRY[name]()


def list_task_verbalizers() -> list[str]:
    return list(_REGISTRY)
