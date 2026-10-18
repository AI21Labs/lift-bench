import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from lift_bench.hierarchical_drafting.pipeline.blocks.utils_blocks import (
    TrainTestSplitBlock,
)


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _run(block: TrainTestSplitBlock) -> None:
    asyncio.run(block.run())


def _make_block(tmp: str, inputs_path: str, **kwargs) -> TrainTestSplitBlock:
    return TrainTestSplitBlock(
        inputs_path=inputs_path,
        output_dir=tmp,
        pipeline_name="test",
        log_level=50,  # CRITICAL — suppress noise
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Hash-split: determinism
# ---------------------------------------------------------------------------


def test_hash_split_is_deterministic():
    """Same records always land in the same split across two independent runs."""
    records = [{"example_id": str(i), "value": i} for i in range(50)]

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)

        block1 = _make_block(tmp, inputs, test_size=0.2, split_by_field="example_id")
        _run(block1)
        train1 = _read_jsonl(str(Path(block1.task_output_dir) / "train.jsonl"))
        test1 = _read_jsonl(str(Path(block1.task_output_dir) / "test.jsonl"))

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)

        block2 = _make_block(tmp, inputs, test_size=0.2, split_by_field="example_id")
        _run(block2)
        train2 = _read_jsonl(str(Path(block2.task_output_dir) / "train.jsonl"))
        test2 = _read_jsonl(str(Path(block2.task_output_dir) / "test.jsonl"))

    assert {r["example_id"] for r in train1} == {r["example_id"] for r in train2}
    assert {r["example_id"] for r in test1} == {r["example_id"] for r in test2}


def test_hash_split_stable_under_upstream_filtering():
    """Dropping records upstream doesn't move surviving records between splits."""
    records = [{"example_id": str(i), "value": i} for i in range(100)]

    with tempfile.TemporaryDirectory() as tmp:
        # Full run
        inputs_full = str(Path(tmp) / "full.jsonl")
        _write_jsonl(inputs_full, records)
        full_block = _make_block(tmp, inputs_full, test_size=0.2, split_by_field="example_id")
        _run(full_block)
        full_test_ids = {r["example_id"] for r in _read_jsonl(str(Path(full_block.task_output_dir) / "test.jsonl"))}
        full_train_ids = {r["example_id"] for r in _read_jsonl(str(Path(full_block.task_output_dir) / "train.jsonl"))}

    with tempfile.TemporaryDirectory() as tmp:
        # Filtered run — drop every other record
        filtered = records[::2]
        inputs_filtered = str(Path(tmp) / "filtered.jsonl")
        _write_jsonl(inputs_filtered, filtered)
        filtered_block = _make_block(tmp, inputs_filtered, test_size=0.2, split_by_field="example_id")
        _run(filtered_block)
        filtered_test_ids = {
            r["example_id"] for r in _read_jsonl(str(Path(filtered_block.task_output_dir) / "test.jsonl"))
        }
        filtered_train_ids = {
            r["example_id"] for r in _read_jsonl(str(Path(filtered_block.task_output_dir) / "train.jsonl"))
        }

    surviving_ids = {r["example_id"] for r in filtered}
    # Every surviving record that was in test/train in the full run is still in the same split
    assert filtered_test_ids == full_test_ids & surviving_ids
    assert filtered_train_ids == full_train_ids & surviving_ids


def test_hash_split_no_leakage():
    """No record appears in both train and test."""
    records = [{"example_id": str(i)} for i in range(200)]

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)
        block = _make_block(tmp, inputs, test_size=0.3, split_by_field="example_id")
        _run(block)
        train_ids = {r["example_id"] for r in _read_jsonl(str(Path(block.task_output_dir) / "train.jsonl"))}
        test_ids = {r["example_id"] for r in _read_jsonl(str(Path(block.task_output_dir) / "test.jsonl"))}

    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {r["example_id"] for r in records}


def test_hash_split_count_based_test_size():
    """Integer test_size / reference_size produces the correct approximate ratio."""
    n = 1000
    records = [{"example_id": str(i)} for i in range(n)]

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)
        # target ~100 test out of 1000
        block = _make_block(tmp, inputs, test_size=100, reference_size=1000, split_by_field="example_id")
        _run(block)
        test_items = _read_jsonl(str(Path(block.task_output_dir) / "test.jsonl"))

    # With 1000 records and a 10% ratio, test count should be within ±5% of 100
    assert 80 <= len(test_items) <= 120


# ---------------------------------------------------------------------------
# Hash-split: split marker written to records
# ---------------------------------------------------------------------------


def test_hash_split_sets_split_field():
    records = [{"example_id": str(i)} for i in range(20)]

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)
        block = _make_block(tmp, inputs, test_size=0.5, split_by_field="example_id")
        _run(block)
        for split_name in ("train", "test"):
            for r in _read_jsonl(str(Path(block.task_output_dir) / f"{split_name}.jsonl")):
                assert r["split"] == split_name


# ---------------------------------------------------------------------------
# Legacy shuffle split still works
# ---------------------------------------------------------------------------


def test_shuffle_split_fallback():
    """split_by_field=None falls back to the legacy sklearn shuffle split."""
    records = [{"example_id": str(i)} for i in range(100)]

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)
        block = _make_block(tmp, inputs, test_size=0.2, split_by_field=None)
        _run(block)
        train = _read_jsonl(str(Path(block.task_output_dir) / "train.jsonl"))
        test = _read_jsonl(str(Path(block.task_output_dir) / "test.jsonl"))

    assert len(train) + len(test) == 100
    assert {r["example_id"] for r in train}.isdisjoint({r["example_id"] for r in test})


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_hash_split_missing_field_raises():
    records = [{"example_id": str(i)} for i in range(10)]

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)
        block = _make_block(tmp, inputs, test_size=0.2, split_by_field="nonexistent_field")
        with pytest.raises(KeyError, match="nonexistent_field"):
            _run(block)


def test_hash_split_count_based_requires_reference_size():
    records = [{"example_id": str(i)} for i in range(10)]

    with tempfile.TemporaryDirectory() as tmp:
        inputs = str(Path(tmp) / "input.jsonl")
        _write_jsonl(inputs, records)
        block = _make_block(tmp, inputs, test_size=5, split_by_field="example_id")  # int test_size, no reference_size
        with pytest.raises(ValueError, match="reference_size"):
            _run(block)
