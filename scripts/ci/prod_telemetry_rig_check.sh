#!/bin/sh
# Read-only check, run ON the serving host, that the container's log and metrics carry what the production rollback rule needs.
#
#   prod_telemetry_rig_check.sh <container> --log-glob GLOB [--expect-levern] [--lines 50000] [--port N] [--container-port 8000]
#                               [--family auto|raw|platform] [--platform-prefix P] [--growth-s 10]
#
# WHERE THE ENGINE LOG IS. A serving container can run with its container log driver off, so `docker logs` is empty. The serving runtime then writes
# the engine's output to a file INSIDE the container (one per engine generation, the newest is the live one); this script reads the tail of the
# newest file that matches --log-glob (a required argument: the glob is the platform's, not this script's) with `docker exec <container> tail`. If the
# glob matches nothing it falls back to `docker logs` (a container started by hand, such as a CI gate, logs there).
#
# It reads and prints; it writes no file and changes nothing: `docker inspect`, `docker port`, `docker exec ... ls/tail/stat/grep/df`, one GET of /metrics
# (inside prod_telemetry.py). Log text never leaves the host: the log goes into prod_telemetry.py check, which prints counts only. Run it with
# traffic in the window (an idle window answers IDLE, exit 3). prod_telemetry.py is found next to this script, at $PROD_TELEMETRY_PY, or (when the host
# has no checkout) as base64 in $PROD_TELEMETRY_B64; $PROD_TELEMETRY_PYBIN names the interpreter (default python3).
#
# The kill-switch section prints which switch files exist and the path QWEN_FAST_LEVERN_OFF_PATH names (when it is set, levern.off is read THERE, not
# in /models/.qwen-c2): the file a runbook writes must be the one the engine reads.
#
# Besides the log lines it checks the LOG FILESYSTEM'S HEADROOM: a log in a small in-memory filesystem fills it, the phase lines stop while the
# engine still works, and every rule that reads them goes blind. It prints the free space, the log's size, its growth over --growth-s seconds
# (0 skips the sample) and the hours to full at that rate. FAIL (exit 2) at 80 % used or under 100 MiB free; a WARN line under 48 h to full.
#
# The port is --port, else the host port Docker published for <container-port>/tcp, else the container port itself.
#
# Exit codes: 0 every required signal present; 2 a required signal absent or the log filesystem nearly full; 3 no traffic in the window;
# 4 the container is not up and healthy, the API does not answer, or the metrics URL does not answer.
set -u

container="${1:-}"
if [ -z "$container" ]; then
  echo "usage: $0 <container> --log-glob GLOB [--expect-levern] [--lines 50000] [--port N] [--container-port 8000] [--family auto|raw|platform] [--platform-prefix P] [--growth-s 10]" >&2
  exit 64
fi
shift
lines=50000
port=""
cport=8000
family=auto
glob=""
expect=""
platform=""
growth_s=10
while [ $# -gt 0 ]; do
  case "$1" in
    --expect-levern) expect="--expect-levern" ;;
    --lines) lines="$2"; shift ;;
    --port) port="$2"; shift ;;
    --container-port) cport="$2"; shift ;;
    --family) family="$2"; shift ;;
    --log-glob) glob="$2"; shift ;;
    --platform-prefix) platform="$2"; shift ;;
    --growth-s) growth_s="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done
if [ -z "$glob" ]; then
  echo "--log-glob is required: the engine log's path glob inside the container" >&2
  exit 64
fi

pybin="${PROD_TELEMETRY_PYBIN:-python3}"
run_py() {
  if [ -n "${PROD_TELEMETRY_B64:-}" ]; then
    "$pybin" -c "$(printf '%s' "$PROD_TELEMETRY_B64" | base64 -d)" "$@"
  else
    "$pybin" "${PROD_TELEMETRY_PY:-$(dirname "$0")/prod_telemetry.py}" "$@"
  fi
}

echo "== container"
state=$(docker inspect -f 'status={{.State.Status}} running={{.State.Running}} restarts={{.RestartCount}} started={{.State.StartedAt}} log_driver={{.HostConfig.LogConfig.Type}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container" 2>&1)
echo "$state"
case "$state" in
  *"running=true"*) ;;
  *) echo "the container is not running: nothing to check yet" >&2; exit 4 ;;
esac
case "$state" in
  *"health=unhealthy"*|*"health=starting"*) echo "the container is not healthy: nothing to check yet" >&2; exit 4 ;;
esac

if [ -z "$port" ]; then
  mapped=$(docker port "$container" "${cport}/tcp" 2>/dev/null | head -1 | sed 's/.*://')
  port="${mapped:-$cport}"
fi
echo "== api (host port $port)"
code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
echo "v1_models_http=${code:-none}"
[ "${code:-}" = "200" ] || { echo "the API does not answer 200: nothing to check yet" >&2; exit 4; }

echo "== kill-switch directory inside the container (names only; absent files are the healthy state)"
docker exec "$container" ls -la /models/.qwen-c2 2>&1 | sed -n '1,20p'
echo "== kill-switch paths the engine reads (names and paths only)"
levern_path=$(docker exec "$container" printenv QWEN_FAST_LEVERN_OFF_PATH 2>/dev/null || true)
echo "QWEN_FAST_LEVERN_OFF_PATH=${levern_path:-<unset: the default /models/.qwen-c2/levern.off>}"
echo "prefix kill switch: /models/.qwen-c2/prefix-reuse.off (fixed in the image; no environment override)"
for name in levern.off prefix-reuse.off; do
  if docker exec "$container" test -e "/models/.qwen-c2/$name" 2>/dev/null; then echo "kill switch PRESENT: $name"; else echo "kill switch absent: $name"; fi
done

echo "== engine log files inside the container (names and sizes)"
newest=$(docker exec "$container" sh -c "ls -t $glob 2>/dev/null | head -1" 2>/dev/null || true)
docker exec "$container" sh -c "ls -la $glob 2>/dev/null" | sed -n '1,10p'

fs_rc=0
echo "== log filesystem headroom"
if [ -n "$newest" ]; then
  df_line=$(docker exec "$container" df -Pk "$newest" 2>/dev/null | tail -1)
  size1=$(docker exec "$container" stat -c %s "$newest" 2>/dev/null || echo 0)
  if [ "$growth_s" -gt 0 ] 2>/dev/null; then sleep "$growth_s"; fi
  size2=$(docker exec "$container" stat -c %s "$newest" 2>/dev/null || echo 0)
  printf '%s\n' "$df_line" | awk -v s1="${size1:-0}" -v s2="${size2:-0}" -v dt="$growth_s" '
    NF >= 6 {
      avail = $4; use = $5 + 0
      rate = (dt > 0 && s2 >= s1) ? (s2 - s1) / dt : 0
      eta = (rate > 0) ? avail * 1024 / rate / 3600 : -1
      printf "log_fs_total_kb=%d avail_kb=%d use_pct=%d log_bytes=%d growth_bytes_per_s=%.1f hours_to_full=%s\n", $2, avail, use, s2, rate, (eta < 0 ? "none" : sprintf("%.1f", eta))
      if (use >= 80 || avail < 102400) { print "FAIL: the log filesystem is nearly full (80 % used or under 100 MiB free): the phase lines will stop and the hang rule goes blind"; bad = 2 }
      else if (eta >= 0 && eta < 48) print "WARN: the log filesystem fills in under 48 hours at the sampled rate"
      seen = 1
    }
    END { if (!seen) print "log_fs=unknown (df gave nothing)"; exit bad }'
  fs_rc=$?
else
  echo "no engine log file matched $glob"
fi

echo "== log and metrics"
metrics="--metrics-url http://127.0.0.1:${port}/metrics --metrics-family $family"
if [ -n "$newest" ]; then
  # boot lines are printed once and scroll out of a tail window: count them over the whole file
  levern_boot=$(docker exec "$container" grep -c -F '[PINDIAG] lever N installed on' "$newest" 2>/dev/null)
  route_boot=$(docker exec "$container" grep -c -F '[PINDIAG] lever N route installed' "$newest" 2>/dev/null)
  echo "log_source=$newest lines=$lines boot_levern_installed=${levern_boot:-0} boot_levern_route=${route_boot:-0}"
  docker exec "$container" tail -n "$lines" "$newest" 2>&1 | run_py check $expect --boot-count "levern_installed=${levern_boot:-0}" --boot-count "levern_route=${route_boot:-0}" $metrics ${platform:+--platform-prefix "$platform"}
  rc=$?
else
  echo "log_source=docker-logs lines=$lines (no file matched $glob)"
  docker logs --tail "$lines" "$container" 2>&1 | run_py check $expect $metrics ${platform:+--platform-prefix "$platform"}
  rc=$?
fi
if [ "$fs_rc" -eq 2 ]; then
  echo "verdict: FAIL (log filesystem nearly full)" >&2
  exit 2
fi
exit "$rc"
