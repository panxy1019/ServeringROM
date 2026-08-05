#!/usr/bin/env bash
set -uo pipefail

ROOT_PID=$1
CONTAINER_ID=$2
RUN_DIR=$3
MAX_SECONDS=${4:-900}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$RUN_DIR"/{samples,stacks,plog-snapshots}
START_EPOCH=$(date +%s)
date -u --iso-8601=seconds >"$RUN_DIR/collector-started-at.txt"

journalctl -kf -o short-iso >"$RUN_DIR/kernel-live.log" 2>&1 &
JOURNAL_PID=$!
trap 'kill "$JOURNAL_PID" 2>/dev/null || true' EXIT

descendants() {
  python3 - "$ROOT_PID" <<'PY'
import sys
from pathlib import Path

pending = [int(sys.argv[1])]
seen = set()
while pending:
    pid = pending.pop()
    if pid in seen:
        continue
    seen.add(pid)
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        continue
    pending.extend(map(int, children))
for pid in sorted(seen):
    print(pid)
PY
}

snapshot_stacks() {
  local stamp=$1
  local pid tid
  {
    for pid in $(descendants); do
      [[ -r /proc/$pid/comm ]] || continue
      printf '===== PID %s COMM %s =====\n' "$pid" "$(cat /proc/$pid/comm)"
      for task in /proc/$pid/task/*; do
        [[ -d $task ]] || continue
        tid=${task##*/}
        printf '%s wchan=' "$tid"
        cat "$task/wchan" 2>/dev/null || true
        printf '\n%s stack:\n' "$tid"
        timeout 1 cat "$task/stack" 2>&1 || true
      done
    done
  } >"$RUN_DIR/stacks/$stamp.txt"
}

iteration=0
while (( $(date +%s) - START_EPOCH < MAX_SECONDS )); do
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  iteration=$((iteration + 1))
  pids=$(descendants | tr '\n' ' ')

  {
    echo "time=$stamp root_pid=$ROOT_PID descendants=$pids"
    for pid in $pids; do
      [[ -r /proc/$pid/comm ]] || continue
      printf '%s ' "$pid"
      cat "/proc/$pid/comm"
      grep -E '^(State|VmPeak|VmSize|VmRSS|Threads|CapEff|CapBnd):' "/proc/$pid/status" 2>/dev/null || true
      printf 'wchan='; cat "/proc/$pid/wchan" 2>/dev/null || true; echo
    done
  } >"$RUN_DIR/samples/process-$stamp.txt"

  if [[ -n ${pids// } ]]; then
    python3 "$SCRIPT_DIR/pidstat_compat.py" $pids --interval 1 \
      >>"$RUN_DIR/pidstat-compat.jsonl" 2>&1 || true
  fi
  npu-smi info >>"$RUN_DIR/npu-smi.log" 2>&1
  printf '\n===== %s =====\n' "$stamp" >>"$RUN_DIR/npu-smi.log"

  if (( iteration % 6 == 0 )); then
    snapshot_stacks "$stamp"
    crictl exec "$CONTAINER_ID" bash -c \
      'find /root/ascend/log/run/plog -type f -printf "%p %s %TY-%Tm-%TdT%TH:%TM:%TS\n" 2>/dev/null | sort' \
      >"$RUN_DIR/plog-snapshots/$stamp.txt" 2>&1 || true
  fi
  sleep 4
done

date -u --iso-8601=seconds >"$RUN_DIR/collector-finished-at.txt"
