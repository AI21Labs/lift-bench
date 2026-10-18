import functools
import logging
from typing import Callable, Optional

from pydantic import ValidationError, validate_call

from lift_bench.hierarchical_drafting.metrics.errors import EvaluationError
from lift_bench.hierarchical_drafting.output_parsers.nested_html import (
    ParsingError,
)
from lift_bench.hierarchical_drafting.output_parsers.registry import (
    PARSER_REGISTRY,
)

logger = logging.getLogger(__name__)

# A check that ran on parsed output scores
# VALID_PARSING_SCORE + (1 - VALID_PARSING_SCORE) * verdict, so a failed check scores this rather than 0.0.
VALID_PARSING_SCORE = 0.00390625


def with_output_parser(metric_func: Callable) -> Callable:
    """Wrap a metric so its registered signature is (completion, parser_name, parser_args, metric_args, **_).

    Arguments are validated against the metric's signature (required, typed, no extras). An unparseable target scores 0.0. An unknown parser, a parser crash or a
    metric that can't run raises EvaluationError.
    """
    validated_metric = validate_call(metric_func)

    @functools.wraps(metric_func)
    async def wrapped(
        completion,
        parser_name: Optional[str] = None,
        parser_args: Optional[dict] = None,
        metric_args: Optional[dict] = None,
        **_,
    ) -> float:
        if parser_name is not None:
            if parser_name not in PARSER_REGISTRY:
                raise EvaluationError(f"unknown output parser '{parser_name}' for metric '{metric_func.__name__}'")
            try:
                parsed = PARSER_REGISTRY[parser_name](completion[-1]["content"], **(parser_args or {}))
            except ParsingError as exc:
                logger.error("output parser '%s' failed for metric '%s': %s", parser_name, metric_func.__name__, exc)
                return 0.0
            except Exception as exc:
                raise EvaluationError(f"output parser '{parser_name}' crashed: {exc}") from exc
            completion = [*completion[:-1], {**completion[-1], "content": parsed}]

        try:
            verdict = await validated_metric(completion, **(metric_args or {}))
        except ParsingError as exc:
            logger.error("metric '%s' found its input unparseable: %s", metric_func.__name__, exc)
            return 0.0
        except ValidationError as exc:
            raise EvaluationError(f"invalid arguments for metric '{metric_func.__name__}': {exc}") from exc
        except Exception as exc:
            raise EvaluationError(f"metric '{metric_func.__name__}' couldn't evaluate: {exc}") from exc
        return VALID_PARSING_SCORE + (1 - VALID_PARSING_SCORE) * float(verdict)

    return wrapped
