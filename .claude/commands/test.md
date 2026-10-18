---
description: Run the test suite
argument-hint: [path] [--coverage]
allowed-tools: [Bash]
---

# Test

Run the project's test suite.

Arguments:

- `path`: Optional path to specific test file or directory
- `--coverage`: Include coverage report

Examples:

- `/test` - Run all tests
- `/test --coverage` - Run all tests with coverage
- `/test tests/lift_bench/test_example.py` - Run specific test file
- `/test tests/lift_bench/` - Run tests in directory

Commands:

- All tests: `inv test`
- With coverage: `inv test --coverage`
- Specific path: `python -m pytest <path> -xvs`
