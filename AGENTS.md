# AI Agent Instructions - Algorithm Research

## Project Overview

This is an ML/algorithm research and experimentation framework. It provides a structured environment for developing, evaluating, and iterating on machine learning models and algorithms, with emphasis on reproducibility, metric tracking, and experiment organization.

### Core Abstractions

- **Models**: Prediction interfaces in `models/` implementing standardized input/output contracts
- **Metrics**: Evaluation functions in `metrics/` for measuring model performance
- **Experiments**: Jupyter notebooks in `notebooks/` for iterative research and analysis
- **Predictions Cache**: Stored model outputs in `cached_model_predictions/` for reproducibility
- **Metrics Cache**: Computed metrics in `cached_prediction_metrics/` for comparison

### Key Capabilities

- Standardized model interface for consistent prediction APIs
- Metric computation and aggregation for model evaluation
- Experiment reproducibility via cached predictions and metrics
- Jupyter notebook-based experimentation workflow
- Data pipeline management with versioned artifacts

---

## Package Structure

```text
lift_bench/
├── __init__.py
├── models/                         # Model implementations
│   └── __init__.py                 # Model interfaces and registry
├── metrics/                        # Evaluation metrics
│   └── __init__.py                 # Metric functions
├── notebooks/                      # Experiment notebooks
│   └── *.ipynb                     # Research experiments
├── data/                           # Input data artifacts
├── cached_model_predictions/       # Stored model outputs
├── cached_prediction_metrics/      # Computed metric results
└── results/                        # Final experiment results

tests/
├── __init__.py
└── lift_bench/
    └── test_*.py                   # Unit tests
```

---

## Development Commands

| Task | Command |
|------|---------|
| Format & lint (incl. import sorting) | `inv formatter --fix; inv lint --fix` |
| Format only | `inv formatter --fix` |
| Lint only | `inv lint` |
| Run tests | `inv test` |
| Run tests with coverage | `inv test --coverage` |
| Type check | `inv staticcheck` |
| Build package | `inv build` |

---

## Code Style & Conventions

### Python Version

- Python 3.14 (`requires-python = ">=3.14,<3.15"`)

### Formatting

- **ruff format**: Line length 120 characters (`ruff.toml`)
- **Import sorting**: ruff's isort rules, combine as imports (`[lint.isort]` in `ruff.toml`)

### Type Hints

Use comprehensive type hints throughout:

```python
from typing import Any, Awaitable, Callable, Optional, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def my_function(cb: Callable[P, Awaitable[R]]) -> ReturnType[P, R]:
    ...
```

### Dataclasses

Prefer frozen dataclasses for immutability:

```python
from dataclasses import dataclass

@dataclass(frozen=True, eq=False, kw_only=True)
class MyClass:
    value: str
    count: int = 0
```

### Pydantic

Use Pydantic for configuration and validation:

```python
from pydantic import BaseModel

class MyConfig(BaseModel):
    timeout: float = 30.0
    max_retries: int = 3
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(x: int, y: str) -> dict:
    """Brief description of function.

    Longer description if needed.

    Args:
        x: Description of x parameter.
        y: Description of y parameter.

    Returns:
        Description of return value.

    Example:
        ```python
        result = my_function(x=1, y="test")
        ```
    """
```

---

## Core Patterns

### Model Interface

Implement models with a consistent prediction interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class BaseModel(ABC, Generic[InputT, OutputT]):
    """Abstract base for all model implementations."""

    @abstractmethod
    def predict(self, inputs: list[InputT]) -> list[OutputT]:
        """Generate predictions for a batch of inputs."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for caching and tracking."""
        ...
```

### Metric Definition

Define metrics as pure functions with clear signatures:

```python
from typing import TypeVar

T = TypeVar("T")

def accuracy(predictions: list[T], ground_truth: list[T]) -> float:
    """Calculate accuracy between predictions and ground truth.

    Args:
        predictions: Model output values.
        ground_truth: Expected correct values.

    Returns:
        Accuracy score between 0.0 and 1.0.
    """
    assert len(predictions) == len(ground_truth)
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)
```

### Caching Pattern

Cache predictions and metrics for reproducibility:

```python
import json
from pathlib import Path

def cache_predictions(model_name: str, predictions: list[dict]) -> Path:
    """Store predictions to cache directory."""
    cache_path = Path(f"cached_model_predictions/{model_name}.jsonl")
    with cache_path.open("w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    return cache_path
```

---

## Testing Conventions

### Test Structure

```text
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures (sample data, mock models)
├── utils.py                    # Test utilities
└── lift_bench/
    ├── __init__.py
    ├── test_models.py          # Model interface tests
    ├── test_metrics.py         # Metric calculation tests
    └── test_pipelines.py       # Data pipeline tests
```

### Model Test Pattern

Test model interfaces with sample data fixtures:

```python
import pytest
from lift_bench.models import MyModel

@pytest.fixture
def sample_inputs() -> list[dict]:
    """Sample input data for model testing."""
    return [{"text": "sample input 1"}, {"text": "sample input 2"}]

def test_model_predict_returns_correct_shape(sample_inputs):
    """Verify model output shape matches input batch size."""
    model = MyModel()
    predictions = model.predict(sample_inputs)
    assert len(predictions) == len(sample_inputs)

def test_model_predict_is_deterministic(sample_inputs):
    """Verify model produces consistent outputs."""
    model = MyModel()
    result1 = model.predict(sample_inputs)
    result2 = model.predict(sample_inputs)
    assert result1 == result2
```

### Test Utilities

Fixtures for research testing patterns:

```python
# conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_data_path(tmp_path) -> Path:
    """Create temporary sample data for testing."""
    data_file = tmp_path / "sample.jsonl"
    data_file.write_text('{"id": 1, "value": "test"}\n')
    return data_file

@pytest.fixture
def mock_model(mocker):
    """Mock model for isolated metric testing."""
    mock = mocker.MagicMock()
    mock.predict.return_value = [{"score": 0.9}, {"score": 0.8}]
    mock.model_name = "mock_model"
    return mock
```

---

## Logging

Prefer using logger with appropriate log level over plain `print`:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Detailed diagnostic info")
logger.info("General operational info")
logger.warning("Something unexpected but handled")
logger.error("Error occurred but app continues")
logger.exception("Error with stack trace")  # Use inside except blocks
```

---

## Key Files Reference

| Category | Files |
|----------|-------|
| **Models** | `lift_bench/models/__init__.py` |
| **Metrics** | `lift_bench/metrics/__init__.py` |
| **Notebooks** | `lift_bench/notebooks/*.ipynb` |
| **Data Artifacts** | `lift_bench/data/`, `cached_model_predictions/`, `cached_prediction_metrics/` |
| **Results** | `lift_bench/results/` |
| **Config** | `pyproject.toml`, `ruff.toml` |
| **CI** | `.pre-commit-config.yaml`, `tasks.py` |
| **Tests** | `tests/lift_bench/` |

---

## Workflow Guidelines

### Commit Workflow

After implementing changes, create commits in this order:

1. Implementation commit(s) - grouped logically (e.g., separate commits for code, tests)
2. Documentation commit (if applicable) - update relevant docs

Use `/commit` to create commits with proper formatting. Do not credit AI assistants in commit messages.

### Documentation

- Update docstrings if public API changed
- Update `docs/` if user-facing behavior changed

### End-of-Session Checklist

At the end of each coding session, consider:

1. **Documentation updates** (`/update_docs`):
   - Docstrings for changed functions/classes
   - docs/ files if user-facing behavior changed

2. **AGENTS.md updates** (`/update_agents`):
   - New patterns or conventions discovered
   - Common mistakes or gotchas encountered
   - New key files created or identified
   - Changes to development workflow

---

## Skills Reference

### format

Format the codebase using the project's formatting tools:

1. Run `inv formatter --fix` to apply ruff formatting (120 char line length)
2. Run `inv lint --fix` to fix ruff lint issues, including import order

Full command: `inv formatter --fix; inv lint --fix`
