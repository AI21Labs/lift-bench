from abc import ABC, abstractmethod
from typing import Optional, Type

from lift_bench.hierarchical_drafting.constraints.base import Constraint
from lift_bench.hierarchical_drafting.constraints.verbalization_mode import (
    SingleTurnVerbalization,
    VerbalizationMode,
)


class ConstraintVerbalizer(ABC):
    def __init__(
        self, supported_constraint: Type[Constraint], verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ):
        self.supported_constraint = supported_constraint
        self.strategies = {"default": self._default_verbalization}
        self.verbalization_mode = verbalization_mode

    def _validate_constraint(self, constraint, supported_constraint: Type[Constraint]) -> None:
        if not isinstance(constraint, supported_constraint):
            raise TypeError(
                f"{self.__class__.__name__} can only verbalize {supported_constraint.__name__}, "
                f"got {constraint.__class__.__name__}"
            )

    def _get_strategy_func(self, strategy: str):
        if strategy not in self.strategies:
            raise KeyError(f"Unknown strategy {strategy} for {self.__class__.__name__}. Have {self.strategies}")
        return self.strategies[strategy]

    def verbalize(
        self, constraint, strategy: str = "default", verbalization_mode: Optional[VerbalizationMode] = None, **kwargs
    ) -> SingleTurnVerbalization:
        self._validate_constraint(constraint=constraint, supported_constraint=self.supported_constraint)
        strategy_func = self._get_strategy_func(strategy)
        if verbalization_mode is None:
            verbalization_mode = self.verbalization_mode
        return strategy_func(constraint, verbalization_mode, **kwargs)

    @abstractmethod
    def _default_verbalization(
        self, constraint: Constraint, verbalization_mode: VerbalizationMode = VerbalizationMode.USER
    ) -> SingleTurnVerbalization:
        pass
