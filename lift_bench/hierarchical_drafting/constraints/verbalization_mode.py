from dataclasses import dataclass
from typing import Optional

from enum import StrEnum


class VerbalizationMode(StrEnum):
    USER = "user"
    COMBINED = "combined"


@dataclass(kw_only=True)
class SingleTurnVerbalization:
    user_verbalization: Optional[str] = None
    system_verbalization: Optional[str] = None

    def to_messages(self) -> list[dict]:
        assert self.user_verbalization is not None, "Must have a non-None user verbalization for converting to messages"

        messages = []
        if self.system_verbalization is not None:
            messages.append({"role": "system", "content": self.system_verbalization})
        messages.append({"role": "user", "content": self.user_verbalization})
        return messages

    def to_prompt(self) -> str:
        assert self.user_verbalization is not None, "Must have a non-None user verbalization for converting to messages"
        assert self.system_verbalization is None, "System message must be None for prompt verbalization"

        return self.user_verbalization

    def __add__(self, other):
        user_verbalization = None
        if self.user_verbalization is not None and other.user_verbalization is not None:
            user_verbalization = self.user_verbalization + "\n" + other.user_verbalization
        elif self.user_verbalization is not None:
            user_verbalization = self.user_verbalization
        elif other.user_verbalization is not None:
            user_verbalization = other.user_verbalization

        system_verbalization = None
        if self.system_verbalization is not None and other.system_verbalization is not None:
            system_verbalization = self.system_verbalization + "\n" + other.system_verbalization
        elif self.system_verbalization is not None:
            system_verbalization = self.system_verbalization
        elif other.system_verbalization is not None:
            system_verbalization = other.system_verbalization

        return SingleTurnVerbalization(
            user_verbalization=user_verbalization,
            system_verbalization=system_verbalization,
        )

    @staticmethod
    def from_string(text, verbalization_mode: VerbalizationMode = VerbalizationMode.USER):
        if verbalization_mode == VerbalizationMode.USER:
            return SingleTurnVerbalization(user_verbalization=text)
        else:
            return SingleTurnVerbalization(system_verbalization=text)


__all__ = ["VerbalizationMode", "SingleTurnVerbalization"]
