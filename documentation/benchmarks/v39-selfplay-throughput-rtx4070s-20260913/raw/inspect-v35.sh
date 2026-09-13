#!/usr/bin/env bash
set -euo pipefail

for name in \
  vast-chess-8gpu-integrated-v35-int8 \
  vast-chess-8gpu-integrated-v35-int8-lr002 \
  vast-chess-8gpu-integrated-v35-int8-from-scratch
do
  log="/workspace/run-control/logs/${name}/supervisor-stdout.log"
  echo "NAME:${name}"
  grep "Completed generation" "${log}" | head -3 || true
  grep "Completed generation" "${log}" | tail -3 || true
done
