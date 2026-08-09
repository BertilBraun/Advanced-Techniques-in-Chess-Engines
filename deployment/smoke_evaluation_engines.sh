#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd -- "${script_directory}/.." && pwd)"
engine_directory="${ENGINE_EVALUATION_DIRECTORY:-${repository_directory}/engines}"

for required_command in grep timeout; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${required_command}" >&2
        exit 1
    fi
done

stockfish_protocol_output="$(printf 'uci\nisready\nquit\n' | "${engine_directory}/stockfish")"
grep -F 'id name Stockfish 18' <<<"${stockfish_protocol_output}"
grep -F 'uciok' <<<"${stockfish_protocol_output}"
grep -F 'readyok' <<<"${stockfish_protocol_output}"

smoke_katago_board() {
    local board_size="$1"
    local request_id="smoke-${board_size}x${board_size}"
    local request
    request="$(printf '{\"id\":\"%s\",\"initialPlayer\":\"B\",\"moves\":[],\"rules\":\"chinese\",\"komi\":7.5,\"boardXSize\":%d,\"boardYSize\":%d,\"maxVisits\":16,\"includePolicy\":true}' "${request_id}" "${board_size}" "${board_size}")"
    local response
    response="$(
        printf '%s\n' "${request}" | timeout 120 \
            "${engine_directory}/katago" analysis \
            -config "${engine_directory}/katago-analysis.cfg" \
            -model "${engine_directory}/katago.bin.gz"
    )"
    grep -Fq "\"id\":\"${request_id}\"" <<<"${response}"
    grep -Eq '"moveInfos":\[\{' <<<"${response}"
    echo "KataGo analysis returned weighted moves for ${board_size}x${board_size}."
}

smoke_katago_board 7
smoke_katago_board 9

echo "Stockfish UCI and KataGo JSON analysis smoke checks passed."
