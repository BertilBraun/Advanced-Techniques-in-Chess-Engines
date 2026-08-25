#!/usr/bin/env bash
# Run one command on a rented compute node over SSH.
#
#   remote_command.sh <destination> <command> [argument ...]
#
# destination: HOST, HOST:PORT, user@HOST or user@HOST:PORT (default user root, default port 22).
# The arguments are joined with spaces and interpreted by the remote login shell, exactly as ssh does,
# so quote anything the local shell must not expand:
#
#   deployment/remote_command.sh 98.142.241.120:22658 nvidia-smi
#   deployment/remote_command.sh 98.142.241.120:22658 'cd /workspace/alphazero-engine && git rev-parse HEAD'
#   deployment/remote_command.sh 98.142.241.120:22658 bash -s < local-script.sh
#
# Key, user, port and the connection options are the only things this script owns; overrides:
# REMOTE_SSH_KEY, REMOTE_SSH_USER, REMOTE_SSH_PORT, REMOTE_SSH_CONNECT_TIMEOUT, REMOTE_SSH_EXTRA_OPTIONS.
# The exit status is the remote command's (255 for connection failures).
#
# The Vast.ai login banner is dropped from stderr so it does not have to be grepped away at every
# call site; stdout is never touched, so binary streams stay intact:
#
#   deployment/remote_command.sh 38.49.42.120:53893 'cd /workspace && tar czf - dir' | tar xzf -
#
# Set REMOTE_SHOW_BANNER=1 to keep the banner.

set -euo pipefail

usage() {
    echo "Usage: remote_command.sh <HOST[:PORT] | user@HOST[:PORT]> <command> [argument ...]" >&2
    exit 2
}

fail() {
    echo "remote_command: $*" >&2
    exit 1
}

default_key() {
    local candidate
    for candidate in "${HOME}/.ssh/codex_vast_ed25519" "${HOME}/.ssh/vast-ssh"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done
    fail "no SSH key found; set REMOTE_SSH_KEY to the local Vast.ai key"
}

[[ $# -ge 2 ]] || usage

destination="$1"
shift

user="${REMOTE_SSH_USER:-root}"
port="${REMOTE_SSH_PORT:-22}"
host="${destination}"
if [[ "${host}" == *@* ]]; then
    user="${host%%@*}"
    host="${host#*@}"
fi
if [[ "${host}" == *:* ]]; then
    port="${host##*:}"
    host="${host%:*}"
fi
[[ -n "${host}" ]] || fail "empty host in destination '${destination}'"
[[ "${port}" =~ ^[0-9]+$ ]] || fail "port must be numeric, got '${port}'"

key="${REMOTE_SSH_KEY:-$(default_key)}"
[[ -f "${key}" ]] || fail "SSH key does not exist: ${key}"

read -r -a extra_options <<< "${REMOTE_SSH_EXTRA_OPTIONS:-}"

ssh_command=(
    ssh
    -o BatchMode=yes
    -o ConnectTimeout="${REMOTE_SSH_CONNECT_TIMEOUT:-25}"
    -o StrictHostKeyChecking=accept-new
    -o IdentitiesOnly=yes
    -i "${key}"
    -p "${port}"
    ${extra_options[@]+"${extra_options[@]}"}
    "${user}@${host}"
    "$@"
)

if [[ -n "${REMOTE_SHOW_BANNER:-}" ]]; then
    exec "${ssh_command[@]}"
fi

# The Vast.ai login banner is written to stderr, so filtering it there leaves stdout byte-exact and
# keeps `tar` and other binary streams usable.
"${ssh_command[@]}" 2> >(grep -v -E \
    '^(Welcome to vast\.ai|Have fun!|AI agents: READ )' >&2)
