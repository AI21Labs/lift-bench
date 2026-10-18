import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from lift_bench.hierarchical_drafting.core.base_blocks import (
    AtomicBlock,
)
from lift_bench.hierarchical_drafting.utils.file_utils import (
    dump_json_lines,
    iter_json_lines,
    makedirs,
)
from sklearn.model_selection import train_test_split

Record = dict[str, Any]
Splits = dict[str, list[Record]]

# Largest value a 128-bit (md5) hash can take, used to map a hash to a fraction in [0, 1).
_MD5_SPACE = 2**128


@dataclass(kw_only=True)
class TrainTestSplitBlock(AtomicBlock):
    version: int = 2
    inputs_path: str
    train_size: float | int = None
    test_size: float | int = None
    stratify_col: str = None
    # If set, assign each record to a split deterministically by hashing item[split_by_field],
    # instead of doing a position-based sklearn shuffle. This makes a record's split depend ONLY
    # on its own key, so it never moves between train/test across runs even when other records are
    # dropped upstream (e.g. generation failures) -- which is required to safely ensemble datasets.
    split_by_field: str = field(default=None, metadata={"relevant": False})
    # Fixed denominator used to turn an integer (count-based) test_size into a stable fraction in
    # hash-split mode: test_ratio = test_size / reference_size. This MUST be a run-invariant count
    # (the initial seed-population size), NOT the number of surviving records -- otherwise the
    # threshold would change between runs and records would shift between splits.
    reference_size: int = field(default=None, metadata={"relevant": False})

    def write_output(self, data: Iterable[dict]) -> None:
        raise NotImplementedError("Outputs should be written through write_outputs()")

    def write_outputs(self, datasets: dict[str, Iterable[dict]]) -> None:
        makedirs(self.task_output_dir)
        for split_name, items in datasets.items():
            split_path = f"{self.task_output_dir}/{split_name}.{self.ext}"
            dump_json_lines(items, split_path)
            self.logger.debug(f"Output written to {split_path}")
        self.logger.info(f"Outputs written to {self.task_output_dir} - splits: {', '.join(list(datasets.keys()))}")

    def _test_ratio(self) -> float:
        # A float test_size in (0, 1) is already a fraction; an integer is a target count that we
        # convert to a fraction of the fixed reference population.
        if isinstance(self.test_size, float) and 0.0 < self.test_size < 1.0:
            ratio = self.test_size
        else:
            if not self.reference_size or self.reference_size <= 0:
                raise ValueError(
                    "reference_size (the fixed initial seed-population size) must be a positive int "
                    "to use a count-based test_size with split_by_field"
                )
            ratio = self.test_size / self.reference_size
        if ratio >= 1.0:
            self.logger.warning(
                f"test_ratio {ratio:.5f} >= 1.0 (test_size={self.test_size}, reference_size={self.reference_size}); "
                f"clamping to 1.0 -- all records will go to the test split"
            )
            ratio = 1.0
        return ratio

    def _hash_frac(self, item: Record) -> float:
        if self.split_by_field not in item:
            raise KeyError(f"split_by_field '{self.split_by_field}' not found in record; cannot compute a stable split")
        digest = hashlib.md5(str(item[self.split_by_field]).encode()).hexdigest()
        return int(digest, 16) / _MD5_SPACE

    def _hash_split(self, items: list[Record]) -> Splits:
        test_ratio = self._test_ratio()
        split_items: Splits = {"train": [], "test": []}
        for item in items:
            # The threshold is fixed across runs, so a given key always lands in the same split.
            # Records assigned to "test" only ever drop out when filtered upstream, so the surviving
            # test set is at most ~test_size (statistically), never more on expectation.
            split_items["test" if self._hash_frac(item) < test_ratio else "train"].append(item)
        if self.train_size is not None:
            self.logger.warning("train_size is ignored in hash-split mode; train is the remainder after test")
        self.logger.info(
            f"Hash split by '{self.split_by_field}': test_ratio={test_ratio:.5f} "
            f"(test_size={self.test_size}, reference_size={self.reference_size}) -> "
            f"{len(split_items['test'])} test / {len(split_items['train'])} train of {len(items)} surviving records"
        )
        return split_items

    def _shuffle_split(self, items: list[Record]) -> Splits:
        stratify = None
        if self.stratify_col is not None:
            stratify = [item[self.stratify_col] for item in items]
        train_items, test_items = train_test_split(
            items,
            train_size=self.train_size,
            test_size=self.test_size,
            random_state=1337,
            shuffle=True,
            stratify=stratify,
        )
        return {"train": train_items, "test": test_items}

    async def _run_impl(self) -> None:
        items = list(iter_json_lines(self.inputs_path))
        split_items: Splits
        if self.split_by_field is not None:
            split_items = self._hash_split(items)
        else:
            split_items = self._shuffle_split(items)

        for curr_split_name, curr_split_items in split_items.items():
            for item in curr_split_items:
                item["split"] = curr_split_name

        self.write_outputs(split_items)
