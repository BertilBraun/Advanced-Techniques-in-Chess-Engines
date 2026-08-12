#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: create_experiment_worktree.sh REVISION DESTINATION" >&2
    exit 2
fi

revision="$1"
destination="$2"
control_repository="${ENGINE_CONTROL_REPOSITORY:-/workspace/alphazero-engine}"

if [[ -e "${destination}" ]]; then
    echo "Worktree destination already exists: ${destination}" >&2
    exit 1
fi

git -C "${control_repository}" fetch origin
git -C "${control_repository}" cat-file -e "${revision}^{commit}"
git -C "${control_repository}" worktree add --detach "${destination}" "${revision}"
"${destination}/deployment/prepare_experiment_worktree.sh"

echo "Prepared ${destination} at $(git -C "${destination}" rev-parse HEAD)."
