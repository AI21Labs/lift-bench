from typing import Dict, Callable

CONSTRAINT2VERBALIZER: Dict[str, Callable] = dict()


def register_constraint_verbalizer(constraint_cls):
    def decorator(verbalizer_cls):
        CONSTRAINT2VERBALIZER[constraint_cls] = verbalizer_cls
        return verbalizer_cls

    return decorator


def get_constraint_verbalizer(constraint):
    for dtype in type(constraint).mro():
        if dtype in CONSTRAINT2VERBALIZER:
            return CONSTRAINT2VERBALIZER[dtype]()
    raise ValueError(f"Unknown constraint type {type(constraint)}")
