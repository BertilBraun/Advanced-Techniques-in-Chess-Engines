#!/usr/bin/env bash
# One-command local quality gate: lint, Python suite, backend suite. Mirrors .github/workflows/checks.yml.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== ruff format =="
(cd "${repository_root}" && ruff format --check .)
echo "== ruff check =="
(cd "${repository_root}" && ruff check .)
echo "== python tests =="
(cd "${repository_root}/py" && python -m pytest --import-mode=importlib ./test -q)
echo "== web backend tests =="
(cd "${repository_root}/deployment/web/backend" && python -m pytest --import-mode=importlib test -q)
echo "All checks passed."
