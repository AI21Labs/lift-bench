#!/usr/bin/env bash

# create .git folder
if [[ ! -d .git ]]; then
  git init
fi

{
  # install uv if not already installed
  uv --version || brew install uv
} && {
  # install the python version in .python-version if missing, create .venv and install the locked dependencies
  uv sync --locked
} && {
  # shellcheck disable=SC1091
  . .venv/bin/activate
} && {
  # install pre-commit hooks (pre-commit itself is a locked dev dependency)
  pre-commit install --install-hooks -t pre-commit -t pre-push -t commit-msg
}
