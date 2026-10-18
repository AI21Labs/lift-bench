---
description: Push current branch to remote
allowed-tools: [Bash]
---

# Push

Push the current branch to the remote repository:

1. Check the current branch name with `git branch --show-current`
2. Check if there are unpushed commits with `git status`
3. If the branch doesn't have an upstream, use `git push -u origin <branch-name>`
4. Otherwise, use `git push`
5. Report the result
