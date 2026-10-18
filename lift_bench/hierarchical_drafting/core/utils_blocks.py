import abc
import bisect
import hashlib
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from lift_bench.hierarchical_drafting.utils.file_utils import (
    dump_json_lines,
    iter_json_lines,
    makedirs,
)

from lift_bench.hierarchical_drafting.core.base_blocks import (
    AtomicBlock,
)


@dataclass(kw_only=True)
class CombineDatasetsBlock(AtomicBlock):
    inputs_paths: str

    async def _run_impl(self) -> None:
        input_paths_list = self.inputs_paths.split(",")
        itr = itertools.chain(*[iter_json_lines(input_path) for input_path in input_paths_list])
        self.write_output(itr)


@dataclass(kw_only=True)
class GenerateSplitsBlock(AtomicBlock):
    version: int = 2
    inputs_path: str
    train_splits: str
    dev_splits: str
    split_by_field: str

    def write_output(self, data: Iterable[dict]) -> None:
        raise NotImplementedError("Outputs should be written through write_outputs()")

    @staticmethod
    def drop_assistant_turn(items: Iterable[dict]) -> Iterable[dict]:
        for item in items:
            if item["messages"][-1]["role"] == "assistant":
                item["messages"] = item["messages"][:-1]
            yield item

    def write_outputs(self, datasets: dict[str, Iterable[dict]]) -> None:
        train_split_names = [split.split("=")[0] for split in self.train_splits.split(",")]
        makedirs(self.task_output_dir)
        for split_name, items in datasets.items():
            split_path = f"{self.task_output_dir}/{split_name}.{self.ext}"
            items_itr = items if split_name in train_split_names else self.drop_assistant_turn(items)
            dump_json_lines(items_itr, split_path)
            self.logger.debug(f"Output written to {split_path}")
        self.logger.info(f"Outputs written to {self.task_output_dir} - splits: {', '.join(list(datasets.keys()))}")

    async def calculate_splits(self) -> list[tuple[str, float]]:
        # parse the input splits string in the format NAME=RATIO,NAME=RATIO, make sure ratios sum to 1
        dev_splits = self.dev_splits.split(",")
        dev_split_names = [split.split("=")[0] for split in dev_splits]
        train_splits = self.train_splits.split(",")
        train_split_names = [split.split("=")[0] for split in train_splits]
        duplicate_splits = set(dev_split_names) & set(train_split_names)
        if duplicate_splits:
            raise ValueError(f"Duplicate splits found: {', '.join(duplicate_splits)}")
        dev_split_ratios = [float(split.split("=")[1]) for split in dev_splits]
        train_split_ratios = [float(split.split("=")[1]) for split in train_splits]
        if round(sum(dev_split_ratios) + sum(train_split_ratios), 3) != 1.0:
            raise ValueError(f"Splits ratios must sum to 1.0, got {sum(dev_split_ratios) + sum(train_split_ratios)}")

        return list(zip(dev_split_names + train_split_names, dev_split_ratios + train_split_ratios))

    def _get_split_for_item(self, item: dict, ratios: list[tuple[str, float]]) -> str:
        if not self.split_by_field:
            item_hash = hashlib.md5(str(item).encode()).hexdigest()
        else:
            item_hash = hashlib.md5(str(item[self.split_by_field]).encode()).hexdigest()
        item_frac = int(item_hash, 16) / 340282366920938463463374607431768211456  # max value for a 128-bit hash
        buckets = np.cumsum([ratio[1] for ratio in ratios])
        buckets = buckets / buckets[-1]
        item_bucket = bisect.bisect_right(buckets, item_frac)  # noqa
        return ratios[item_bucket][0]

    async def _run_impl(self) -> None:
        splits_ratios = await self.calculate_splits()
        items = list(iter_json_lines(self.inputs_path))
        split_items = {}
        for item in items:
            split_name = self._get_split_for_item(item, splits_ratios)
            if split_name not in split_items:
                split_items[split_name] = []
            item["split"] = split_name
            split_items[split_name].append(item)

        self.write_outputs(split_items)


@dataclass(kw_only=True)
class ExplodeFieldBlock(AtomicBlock):
    inputs_path: str
    field_name: str
    keep_original_list: bool = False
    new_id_field: str = ""

    async def _run_impl(self) -> None:
        items = []
        for item in iter_json_lines(self.inputs_path):
            if self.field_name not in item:
                self.logger.error(f"Field '{self.field_name}' not found in item: {item}")
                raise KeyError(f"Field '{self.field_name}' not found. Cannot continue.")
            if not isinstance(item[self.field_name], list):
                self.logger.error(f"Field '{self.field_name}' is not a list in item: {item}")
                raise ValueError(f"Field '{self.field_name}' should be a list. Cannot continue.")
            for value in item[self.field_name]:
                new_item = item.copy()
                if self.keep_original_list:
                    new_item[f"{self.field_name}_list"] = item[self.field_name]
                new_item[self.field_name] = value
                if self.new_id_field:
                    new_item[self.new_id_field] = hashlib.md5(str(value).encode()).hexdigest()
                items.append(new_item)
        self.write_output(items)


@dataclass(kw_only=True)
class AggregateBlock(AtomicBlock):
    inputs_path: str
    group_field_names: list[str]
    agg_field_names: list[str] = None

    @abc.abstractmethod
    async def _aggregate_items(self, items: list[dict]) -> dict: ...

    def _validate_keys(self, item: dict, keys: list[str]) -> None:
        for k in keys:
            if k not in item:
                self.logger.error(f"Key '{k}' not found in item: {item}")
                raise KeyError(f"Key '{k}' not found. Cannot continue.")

    async def _run_impl(self) -> None:
        grouped = defaultdict(list)
        for item in iter_json_lines(self.inputs_path):
            # check that group keys are defined
            self._validate_keys(item, self.group_field_names)
            group_key = tuple(item[k] for k in self.group_field_names)
            # if defined agg fields, check they are present
            if self.agg_field_names is not None and len(self.agg_field_names) > 0:
                self._validate_keys(item, self.agg_field_names)
                item = {k: v for k, v in item.items() if k in self.agg_field_names}
            grouped[group_key].append(item)

        # apply _aggregate_items on each group
        grouped_items = []
        for group_key, group_rows in grouped.items():
            agg_out = await self._aggregate_items(group_rows)
            # add group keys to each aggregation result
            key_dict = dict(zip(self.group_field_names, group_key))
            combined = {**key_dict, **agg_out}
            grouped_items.append(combined)
        self.write_output(grouped_items)


@dataclass(kw_only=True)
class DropDuplicatesBlock(AtomicBlock):
    version: int = 2
    inputs_path: str
    field_names: str

    async def _run_impl(self) -> None:
        field_names = [fn.strip() for fn in self.field_names.split(",")]
        seen = set()
        unique_items = []
        counter = 0
        for item in iter_json_lines(self.inputs_path):
            counter += 1
            key = tuple(str(item[field]) for field in field_names)
            if key in seen:
                continue
            seen.add(key)
            unique_items.append(item)

        self.logger.debug(
            f"Processed {counter} items, found {len(unique_items)} unique items by fields: {', '.join(field_names)}"
        )
        self.write_output(unique_items)


@dataclass(kw_only=True)
class HeadBlock(AtomicBlock):
    inputs_path: str
    n_items: int

    async def _run_impl(self) -> None:
        items = []
        for counter, item in enumerate(iter_json_lines(self.inputs_path)):
            if counter >= self.n_items:
                break
            items.append(item)
            counter += 1

        self.logger.debug(f"Returning first {len(items)} items (requested {self.n_items})")
        self.write_output(items)


@dataclass(kw_only=True)
class PartitionByFieldBlock(AtomicBlock):
    version: int = 2
    inputs_path: str
    field_name: str
    allow_missing_field: bool = True
    partition_paths: dict[str, str] = field(default_factory=dict, metadata={"relevant": False})

    _PARTITIONS_FILE_NAME: str = field(default="_partition_names", metadata={"relevant": False})

    def run_on_skipped(self) -> None:
        partitions_file = f"{self.task_output_dir}/{self._PARTITIONS_FILE_NAME}.{self.ext}"
        for partition in iter_json_lines(partitions_file):
            partition_value = partition["partition"]
            output_path = f"{self.task_output_dir}/{partition_value}.{self.ext}"
            self.partition_paths[partition_value] = output_path

    async def _run_impl(self) -> None:
        partitioned_items = {}
        for item in iter_json_lines(self.inputs_path):
            field_value = item.get(self.field_name, None)
            if not self.allow_missing_field and field_value is None:
                self.logger.error(f"Field '{self.field_name}' not found in item: {item}")
                raise KeyError(f"Field '{self.field_name}' not found. Cannot continue.")
            if field_value not in partitioned_items:
                partitioned_items[field_value] = []
            partitioned_items[field_value].append(item)

        partitioned_items[self._PARTITIONS_FILE_NAME] = [{"partition": k} for k in list(partitioned_items.keys())]
        for partition_value, items in partitioned_items.items():
            output_path = f"{self.task_output_dir}/{partition_value}.{self.ext}"
            self.partition_paths[partition_value] = output_path
            dump_json_lines(items, output_path)
            if partition_value != self._PARTITIONS_FILE_NAME:
                self.logger.debug(f"Wrote {len(items)} items to partition file: {output_path}")

        self.logger.info(f"Wrote {len(partitioned_items) - 1} partitions to {self.task_output_dir}")
