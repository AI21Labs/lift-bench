---
description: Create a git commit with conventional format and gitmoji
allowed-tools: [Bash, Read, Glob, Grep]
---

# Commit

Create a git commit following these steps:

1. Run `git status` and `git diff --staged` to see what's being committed
2. If nothing is staged, run `git diff` to see unstaged changes and suggest what to stage
3. Analyze the changes and generate a commit message:
   - Use conventional commit format: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`
   - Add a relevant gitmoji after the prefix (e.g., `feat: ✨ add new feature`)
   - Keep the message concise and descriptive
   - Do not credit yourself in the message
4. Stage the relevant files (prefer specific files over `git add -A`)
5. Create the commit
6. Show the result with `git log -1`

Common gitmojis:

- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation
- ♻️ Refactor
- ✅ Tests
- 🔧 Configuration
- 🚀 Performance
- 🔒 Security
