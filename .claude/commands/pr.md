---
description: Create a pull request for the current branch
argument-hint: [title]
allowed-tools: [Bash, Read]
---

# Pull Request

Create a pull request for the current branch:

1. Verify all changes are committed: `git status`
2. Ensure branch is pushed to remote: `git push -u origin $(git branch --show-current)` if needed
3. Get the list of commits on this branch: `git log origin/master..HEAD --oneline`
4. Generate PR description from commits
5. Open the PR on GitHub: `gh pr create --title "<title>" --body "<description>"`

Return the PR URL when done.
