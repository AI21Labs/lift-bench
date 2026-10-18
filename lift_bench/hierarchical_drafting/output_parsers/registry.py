from typing import Callable

PARSER_REGISTRY: dict[str, Callable[[str], str]] = {}


def register_parser(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        PARSER_REGISTRY[name] = func
        return func

    return decorator
