#!/usr/bin/env bash
set -euo pipefail

build_directory="${1:-build-clang}"

find src test -type f -name '*.cpp' -print0 |
    xargs -0 clang-tidy -p "${build_directory}"
