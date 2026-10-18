from dataclasses import dataclass
from typing import Any

from lift_bench.hierarchical_drafting.core.base_blocks import (
    MapperBlock,
)
from lift_bench.hierarchical_drafting.document.document_factory import (
    HeaderFactory,
)


@dataclass(kw_only=True)
class DocumentOutlineTemplateBlock(MapperBlock):
    max_children: int
    max_depth: int
    template_verbalizer_name: str

    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict:
        if item.get("error", None) is not None:
            return item

        document = HeaderFactory.generate(self.max_children, self.max_depth)
        item["document_template"] = document.to_md()
        return item
