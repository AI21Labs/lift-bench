import abc
import hashlib
import json
import logging
import math
from dataclasses import dataclass, fields, field
from typing import Any, Iterable

import more_itertools
from lift_bench.hierarchical_drafting.utils.file_utils import (
    dump_json_lines,
    iter_json_lines,
    check_path_exists,
    makedirs,
    dump_json,
    dump_text,
    rmfile,
    load_yaml,
)

from lift_bench.hierarchical_drafting.utils.async_utils import (
    gather_with_concurrency,
)

DEFAULT_OUTPUT_DIR = "."

logging.basicConfig(level=logging.INFO)


@dataclass(kw_only=True)
class AtomicBlock(abc.ABC):
    version: int = 1
    output_dir: str = field(default=DEFAULT_OUTPUT_DIR, metadata={"relevant": False})
    description: str | None = field(default=None, metadata={"relevant": False})
    max_concurrency: int = field(default=1, metadata={"relevant": False})
    write_params: bool = field(default=True, metadata={"relevant": False})
    pipeline_name: str = field(default="", metadata={"relevant": False})
    disable_params_hash: bool = field(default=False, metadata={"relevant": False})
    partitions: list[tuple[str, str]] = field(default_factory=list, metadata={"relevant": False})
    logger_name: str = field(default="", metadata={"relevant": False})
    log_level: int = field(default=logging.INFO, metadata={"relevant": False})
    dry_run: bool = field(default=False, metadata={"relevant": False})

    # TODO reinstate? async issue...
    # def __post_init__(self):
    #     self.run()

    @property
    def logger(self) -> logging.Logger:
        if self.logger_name:
            logger_name = self.logger_name
        elif self.pipeline_name:
            logger_name = f"{self.pipeline_name}"
        else:
            logger_name = "alignment"
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        return logger

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def ext(self) -> str:
        return "jsonl"

    @property
    def output_filename(self) -> str:
        return "output"

    @property
    def error_filename(self) -> str:
        return "error"

    def get_attributes_dict(self, relevant_only: bool) -> dict:
        output_fields = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("relevant", True) or (not relevant_only)
        }
        return output_fields

    @property
    def attributes_hash(self) -> str:
        relevant_fields = self.get_attributes_dict(relevant_only=True)
        sorted_json = json.dumps(relevant_fields, sort_keys=True)
        return hashlib.md5(sorted_json.encode()).hexdigest()

    @property
    def partitions_infix(self) -> str:
        if self.partitions:
            return "/" + ("/".join([f"{k}={v}" for k, v in self.partitions]))
        return ""

    @property
    def task_output_dir(self) -> str:
        pipeline_dir_path = self.output_dir
        if self.pipeline_name:
            pipeline_dir_path = f"{pipeline_dir_path}/{self.pipeline_name}"
        task_dir = self.name
        if not self.disable_params_hash:
            task_dir += f"_{self.attributes_hash}"
        task_dir += self.partitions_infix
        return f"{pipeline_dir_path}/{task_dir}"

    @property
    def output_path(self) -> str:
        return f"{self.task_output_dir}/{self.output_filename}.{self.ext}"

    @property
    def error_path(self) -> str:
        return f"{self.task_output_dir}/{self.error_filename}.{self.ext}"

    @property
    def success_marker_path(self) -> str:
        return f"{self.task_output_dir}/_SUCCESS"

    @property
    def params_path(self) -> str:
        return f"{self.task_output_dir}/_params.json"

    def is_complete(self) -> bool:
        return check_path_exists(self.success_marker_path)

    def write_output(self, data: Iterable[dict]) -> None:
        makedirs(self.task_output_dir)
        dump_json_lines(data, self.output_path)
        self.logger.info(f"Output written to {self.output_path}")

    def write_error(self, data: Iterable[dict]) -> None:
        makedirs(self.task_output_dir)
        dump_json_lines(data, self.error_path)
        self.logger.info(f"Error written to {self.error_path}")

    def write_success_marker(self) -> None:
        dump_text("", self.success_marker_path, exist_ok=False)
        self.logger.debug(f"Success marker written to {self.success_marker_path}")

    def dry_run_action(self) -> None:
        attributes = json.dumps(self.get_attributes_dict(relevant_only=True), indent=4)
        output_str = f"\n{self.name}\n{self.output_path}\n{attributes}\n"
        self.logger.info(output_str)

    def run_on_skipped(self) -> None: ...

    async def run(self) -> None:
        if self.dry_run:
            self.dry_run_action()
            return
        if self.is_complete():
            self.run_on_skipped()
            self.logger.debug(f"Found success marker at {self.success_marker_path}")
            self.logger.info(f"Output already exists at {self.task_output_dir} for {self.name} - skipping.")
            return

        self.logger.info(f"Running {self.name}...")
        self.logger.info(f"Output directory: {self.task_output_dir}")
        makedirs(self.task_output_dir)
        if self.write_params:
            dump_json(self.get_attributes_dict(relevant_only=False), self.params_path, sort_keys=True, indent=4)

        await self._run_impl()
        self.write_success_marker()
        self.logger.info(f"Done running {self.name}.")

    @abc.abstractmethod
    async def _run_impl(self) -> None: ...


@dataclass(kw_only=True)
class MapperBlockBase(AtomicBlock, abc.ABC):
    """A block that maps over the input data and applies a function to each item."""

    inputs_path: str
    min_success_threshold: float = 0.0
    chunk_size: int = field(default=1000, metadata={"relevant": False})

    def preprocess(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def is_error(item: dict) -> bool:
        return item.get("error", None) is not None

    def get_iterator(self) -> Iterable[dict]:
        return iter_json_lines(self.inputs_path)

    def batch_path(self, idx: int) -> str:
        return f"{self.task_output_dir}/_output_{self.chunk_size}_{idx:05}.{self.ext}"

    def write_partial_output(self, items: list[dict], idx: int) -> None:
        out_path = self.batch_path(idx)
        dump_json_lines(items, out_path)
        self.logger.debug(f"Partial output written to {out_path}")

    def batch_output_exists(self, batch_id: int) -> bool:
        return check_path_exists(self.batch_path(batch_id))

    def combine_batches(self, batch_paths: list[str]) -> None:
        items: list[dict] = []
        for batch_path in batch_paths:
            items.extend(list(iter_json_lines(batch_path)))
        output_items = [item for item in items if not self.is_error(item)]
        error_items = [item for item in items if self.is_error(item)]
        if output_items:
            self.write_output(output_items)
        if error_items:
            self.write_error(error_items)
        for batch_path in batch_paths:
            rmfile(batch_path)
        if len(output_items) < self.min_success_threshold * len(items):
            error_message = (
                f"Not enough successful items: {len(output_items)} out of {len(items)}"
                f"(threshold: {self.min_success_threshold}, {self.min_success_threshold * len(items):.2f})"
            )
            self.logger.error(error_message)
            raise RuntimeError(error_message)

    @abc.abstractmethod
    async def _run_impl(self): ...


@dataclass(kw_only=True)
class BatchMapperBlock(MapperBlockBase, abc.ABC):
    batch_size: int = field(default=1, metadata={"relevant": True})

    @abc.abstractmethod
    async def process_items(self, items: list[dict], prep_data: dict[str, Any]) -> list[dict]: ...

    async def _run_impl(self):
        prep_data = self.preprocess()
        # TODO this should either be a generator or computed for each chunk separately
        processing_futures = [
            self.process_items(items, prep_data)
            for items in more_itertools.chunked(self.get_iterator(), self.batch_size)
        ]
        batch_paths = []
        batches_per_chunk = self.chunk_size // self.batch_size
        actual_chunk_size = batches_per_chunk
        for batch_id, futures_partition in enumerate(more_itertools.chunked(processing_futures, batches_per_chunk)):
            batch_paths.append(self.batch_path(batch_id))
            if self.batch_output_exists(batch_id):
                self.logger.info(f"Batch {batch_id} already exists - skipping.")
                continue

            total_batches = len(processing_futures)
            batch_start = actual_chunk_size * batch_id
            next_batch_start = actual_chunk_size * (batch_id + 1)
            item_min = batch_start * self.batch_size
            item_max = next_batch_start * self.batch_size - 1
            tqdm_kwargs = {
                "total": total_batches,
                "desc": f"Processing {self.name} batch {batch_id + 1: 2}/{math.ceil(total_batches / batches_per_chunk)}",
                "initial": batch_start,
                "postfix": {"batch_range": f"{item_min}-{item_max}"},
            }
            # Process the items in the batch
            batches = await gather_with_concurrency(
                self.max_concurrency,
                *futures_partition,
                log_name=self.logger.name,
                use_tqdm=True,
                tqdm_kwargs=tqdm_kwargs,
            )
            items = [item for batch in batches for item in batch]
            # Write the batch output
            self.write_partial_output(items, batch_id)

        self.combine_batches(batch_paths)


@dataclass(kw_only=True)
class MapperBlock(MapperBlockBase, abc.ABC):
    @abc.abstractmethod
    async def process_item(self, item: dict, prep_data: dict[str, Any]) -> dict: ...

    async def _run_impl(self):
        prep_data = self.preprocess()
        # TODO this should either be a generator or computed for each chunk separately
        processing_futures = [self.process_item(item, prep_data) for item in self.get_iterator()]
        batch_paths = []
        for batch_id, futures_partition in enumerate(more_itertools.chunked(processing_futures, self.chunk_size)):
            batch_paths.append(self.batch_path(batch_id))
            if self.batch_output_exists(batch_id):
                self.logger.info(f"Batch {batch_id} already exists - skipping.")
                continue

            total_items = len(processing_futures)
            batch_min = batch_id * self.chunk_size
            batch_max = min((batch_id + 1) * self.chunk_size - 1, total_items)
            tqdm_kwargs = {
                "total": total_items,
                "desc": f"Processing {self.name} batch {batch_id + 1: 2}/{total_items // self.chunk_size + 1}",
                "initial": batch_id * self.chunk_size,
                "postfix": {"batch_range": f"{batch_min}-{batch_max}"},
            }
            # Process the items in the batch
            items = await gather_with_concurrency(
                self.max_concurrency,
                *futures_partition,
                log_name=self.logger.name,
                use_tqdm=True,
                tqdm_kwargs=tqdm_kwargs,
            )
            # Write the batch output
            self.write_partial_output(items, batch_id)

        self.combine_batches(batch_paths)


@dataclass(kw_only=True)
class FilterBlock(AtomicBlock, abc.ABC):
    inputs_path: str

    def preprocess(self) -> dict[str, Any]:
        return {}

    def get_iterator(self) -> Iterable[dict]:
        return iter_json_lines(self.inputs_path)

    @abc.abstractmethod
    def filter_item(self, item: dict, prep_data: dict[str, Any]) -> bool: ...

    async def _run_impl(self) -> None:
        prep_data = self.preprocess()
        items = [item if self.filter_item(item, prep_data) else None for item in self.get_iterator()]
        orig_len = len(items)
        items = [item for item in items if item is not None]
        filt_len = len(items)
        self.logger.info(f"Keeping {filt_len} items out of {orig_len} ({(filt_len / orig_len) * 100:.2f}%)")
        self.write_output(items)


@dataclass(kw_only=True)
class ConfigBlock(AtomicBlock, abc.ABC):
    """A block that reads configuration from a file or other source."""

    config_path: str
    _configs: dict = field(init=False, repr=False, metadata={"relevant": False})
    config_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        if not check_path_exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        self._configs = load_yaml(self.config_path)
        self.config_hash = hashlib.md5(json.dumps(self._configs).encode()).hexdigest()
