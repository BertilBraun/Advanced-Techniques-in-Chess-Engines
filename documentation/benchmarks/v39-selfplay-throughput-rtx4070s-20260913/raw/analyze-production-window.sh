#!/usr/bin/env bash
set -euo pipefail

log_path=/workspace/run-control/logs/vast-chess-8gpu-progressive-v39-int8/supervisor-stdout.log
awk '
    /Completed generation 20:/ { active = 1; next }
    /Completed generation 34:/ { active = 0 }
    active && /Appended [0-9]+ staged games and [0-9]+ samples/ {
        games += $4
        samples += $8
        append_seconds_text = $11
        sub(/s$/, "", append_seconds_text)
        append_seconds += append_seconds_text
        batches += 1
    }
    END {
        printf "games=%d samples=%d batches=%d append_seconds=%.1f append_positions_per_s=%.1f positions_per_s=%.3f games_per_h=%.1f avg_positions_per_game=%.3f\n", \
            games, samples, batches, append_seconds, samples / append_seconds, samples / 1680, games * 3600 / 1680, samples / games
    }
' "${log_path}"
