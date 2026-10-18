---
description: Update documentation based on code changes
argument-hint: [file-or-module]
allowed-tools: [Read, Edit, Glob, Grep, Bash]
---

# Update Docs

Update documentation to reflect code changes:

1. **If a specific file/module is provided**, focus on that area
2. **Otherwise**, check recent git changes: `git diff --name-only HEAD~5` to identify what changed

For each changed file, update relevant documentation:

## Docstrings

- Update function/class docstrings if signatures or behavior changed
- Use Google-style docstrings with Args, Returns, Example sections
- Keep docstrings concise but complete

## docs/ folder

- Update relevant `.md` files in `docs/` if user-facing behavior changed
- Check if new public API needs documentation

## Type hints

- Ensure type hints match actual behavior
- Add missing type hints to public functions

## What NOT to update

- CHANGELOG (maintained externally)
- AGENTS.md (use `/update_agents` for that)

After updates, run `pre-commit run markdownlint --files <changed .md files>` to lint the edited docs.
