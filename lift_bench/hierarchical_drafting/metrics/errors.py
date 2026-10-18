"""How a check that can't produce a verdict fails.

- An unparseable target (the output parser raises ParsingError, or a metric does) scores 0.0.
- Anything else that stops a check -- an unknown parser, a parser crash, invalid arguments, MetricError, an
  unexpected exception -- is an evaluation error. The other checks still run, but the task gets no score:
  constraint_reward raises EvaluationError.
"""

from typing import Any, Optional


class MetricError(Exception):
    """A metric couldn't evaluate (bad argument values, a failed judge call)."""


class EvaluationError(Exception):
    """A task's checks couldn't all be evaluated, so the task has no score.

    `agg_eval_scores` holds every check's score, None for the ones that errored.
    """

    def __init__(
        self,
        message: str,
        agg_eval_scores: Optional[list[Optional[float]]] = None,
        agg_eval_ids: Optional[list[Any]] = None,
    ):
        super().__init__(message)
        self.agg_eval_scores = agg_eval_scores or []
        self.agg_eval_ids = agg_eval_ids
