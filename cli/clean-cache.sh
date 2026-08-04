#!/bin/sh

# Remove local Python tooling caches from the repository.

set -eu

script_directory="$(cd -- "$(dirname -- "$0")" && pwd)"
project_root="$(dirname -- "$script_directory")"

cd -- "$project_root"

find . \
  \( -path './.git' -o -path './.venv' \) -prune -o \
  -type d \
  \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' \) \
  -prune -print -exec rm -rf -- {} +
