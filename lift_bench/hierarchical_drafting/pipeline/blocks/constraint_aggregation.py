from lift_bench.hierarchical_drafting.core.utils_blocks import (
    AggregateBlock,
)


class ConstraintTypeAggregationBlock(AggregateBlock):
    async def _aggregate_items(self, items: list[dict]) -> dict:
        result_dict = {}
        for item in items:
            constraint_dct = item["constraint_dct"]
            constraint_name = item["constraint_name"]
            constraint_type = item["constraint_type"]
            if constraint_type not in result_dict:
                result_dict[constraint_type] = {}
            for header_name, constraint_lst in constraint_dct.items():
                if header_name not in result_dict[constraint_type]:
                    result_dict[constraint_type][header_name] = {}
                result_dict[constraint_type][header_name][constraint_name] = constraint_lst
        return result_dict
