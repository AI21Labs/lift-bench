---
description: Review changes on the current branch
allowed-tools: [Bash, Read, Glob, Grep]
---

# Review Pull Request

Review the changes on the current branch compared to master:

1. Get branch info and changed files:

   ```bash
   git log origin/master..HEAD --oneline
   git diff origin/master...HEAD --stat
   ```

2. Get the full diff:

   ```bash
   git diff origin/master...HEAD
   ```

3. Review the changes for:
   - Code correctness and potential bugs
   - Adherence to project conventions (see AGENTS.md)
   - Type hints and documentation
   - Test coverage for new functionality
   - Security concerns (hardcoded secrets, injection vulnerabilities)
   - Performance implications

4. Read specific files if deeper context is needed

5. Provide a structured review with:
   - **Summary**: Brief overview of what the changes do
   - **Strengths**: What's done well
   - **Concerns**: Issues that should be addressed
   - **Suggestions**: Optional improvements
   - **Verdict**: Approve, Request Changes, or Comment
