---
description: Run static type checking with mypy
allowed-tools: [Bash]
---

# Typecheck

Run static type checking on the codebase:

```bash
inv staticcheck
```

This runs mypy on:

- `lift_bench/`
- `tests/`

Report any type errors found and suggest fixes if appropriate.
