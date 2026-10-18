from abc import ABC, abstractmethod


class TaskVerbalizer(ABC):
    @abstractmethod
    async def build_messages(
        self,
        topic: str,
        document_outline: str,
        header_to_constraints_dct: dict,
        use_strict_header_names: bool,
        verbalized_topic: str | None = None,
    ) -> list[dict]: ...
