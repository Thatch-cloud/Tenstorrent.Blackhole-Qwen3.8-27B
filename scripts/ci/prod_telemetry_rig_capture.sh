#!/bin/sh
# Baseline and production capture, run ON the serving host: follows the container's engine log and feeds it to prod_telemetry.py capture, which
# writes derived records only (round times with the host load, request shapes, events, triggers) to a directory on this host. Nothing is sent anywhere,
# and no raw log line is written.
#
#   prod_telemetry_rig_capture.sh <container> <out-dir> --log-glob GLOB [--port N] [--container-port 8000] [--platform-prefix P] [--prom-file F]
#                                 [--maintenance-file F]
#
# Run it as a service (a systemd --user unit with Restart=always: a crash or a reboot must not end the only exact hang rule) or, for a quick look, under
# nohup in its own session (`setsid nohup ... &`) so one `kill -TERM -- -<pgid>` stops everything it started. It
# exits when stopped (SIGTERM or SIGINT: it stops the follower, closes the files and removes its pid file). It refuses to start (exit 75) while another
# capture into the same <out-dir> is alive: two would write every record twice. It survives the container restarting (it waits for the container and
# picks up the newest engine log of the new run) and the engine log rolling to a new generation inside one container (it re-follows within 30 s).
# Each time it attaches it prints a follow marker line carrying the log file's name, the container's image id and the container id, which the capture
# turns into a `follow` event: the capture directory records which image produced which records.
#
# --prom-file F: the capture writes its heartbeat and a counter per trigger code to F as Prometheus text (point it into the host's node_exporter textfile
# directory, so a log-only trigger and a dead capture reach the alert rules). --maintenance-file F: while F exists (and is under 12 h old) the Tier 0 triggers
# are recorded as suppressed instead of raised: planned window work restarts the engine, and touching F before it starts is the supported way to say so.
#
# Triggers (the watcher's Tier 0 and Tier 1 reports) print on its stdout and append to <out-dir>/triggers.jsonl; it never writes a kill switch or
# restarts anything. The hang rule's "a request is live" is read from the engine's /metrics gauges on 127.0.0.1:<port> every 10 s when the engine
# exports them, else from the last stats line in the log (see prod_telemetry.py). The port is --port, else the host port Docker published for
# <container-port>/tcp, else the container port itself; unless --port fixes it, the follower looks the published port up again on every attach and every
# recheck and writes the URL to <out-dir>/metrics.url, which the poller re-reads: a redeploy that moves the port does not end the metrics poll (an ended
# poll would read as an unreachable API). prod_telemetry.py is found next to this script, at $PROD_TELEMETRY_PY, or as base64 in
# $PROD_TELEMETRY_B64; $PROD_TELEMETRY_PYBIN names the interpreter (default python3). $PROD_TELEMETRY_RECHECK_S (30) and $PROD_TELEMETRY_RETRY_S (5) set
# the follower's polling.
#
# The engine log is a file INSIDE the container (see prod_telemetry_rig_check.sh). `docker exec ... tail -F` leaves its remote tail running when the
# docker client is killed, so the follower also kills that tail inside the container when it stops or re-follows (a tail left over only reads, and it
# ends itself at its next write to the closed pipe).
set -u

pybin="${PROD_TELEMETRY_PYBIN:-python3}"
recheck="${PROD_TELEMETRY_RECHECK_S:-30}"
retry="${PROD_TELEMETRY_RETRY_S:-5}"

if [ "${1:-}" = "--follower" ]; then
  # internal: the follow loop, as its own process so that it can be stopped by pid
  shift
  container="$1"; glob="$2"; followpid="$3"; urlfile="${4:--}"; cport="${5:-8000}"
  tailpid=""
  curfile=""

  newest() {
    docker exec "$container" sh -c "ls -t $glob 2>/dev/null | head -1" 2>/dev/null
  }

  kill_remote_tail() {
    [ -n "$1" ] || return 0
    script='for d in /proc/[0-9]*; do p=${d#/proc/}; c=$(tr "\0" " " < $d/cmdline 2>/dev/null); case "$c" in "tail -n 0 -F '"$1"'"*) kill $p 2>/dev/null;; esac; done'
    docker exec "$container" sh -c "$script" >/dev/null 2>&1 || true
  }

  stop_tail() {
    if [ -n "$tailpid" ]; then
      kill "$tailpid" 2>/dev/null
      wait "$tailpid" 2>/dev/null
    fi
    kill_remote_tail "$curfile"
    tailpid=""
    curfile=""
  }

  publish_url() {
    # the engine's published port as of now, for the poller (skipped when --port fixed it: urlfile is -)
    [ "$urlfile" != "-" ] || return 0
    mapped=$(docker port "$container" "${cport}/tcp" 2>/dev/null | head -1 | sed 's/.*://')
    if [ -n "$mapped" ]; then
      echo "http://127.0.0.1:${mapped}/metrics" > "$urlfile.tmp" 2>/dev/null && mv -f "$urlfile.tmp" "$urlfile"
    fi
    return 0
  }

  trap 'stop_tail; rm -f "$followpid"; exit 143' TERM INT HUP
  echo $$ > "$followpid"
  while :; do
    file=$(newest)
    if [ -z "$file" ]; then
      sleep "$retry" &
      wait $!
      continue
    fi
    image=$(docker inspect -f '{{.Image}}' "$container" 2>/dev/null | cut -c1-19)
    cid=$(docker inspect -f '{{.Id}}' "$container" 2>/dev/null | cut -c1-12)
    publish_url
    echo "[prod-telemetry] follow $file image=${image:-unknown} container=${cid:-unknown}"
    docker exec "$container" tail -n 0 -F "$file" 2>/dev/null &
    tailpid=$!
    curfile="$file"
    while kill -0 "$tailpid" 2>/dev/null; do
      sleep "$recheck" &
      wait $!
      publish_url
      [ "$(newest)" = "$file" ] || break
    done
    stop_tail
    sleep "$retry" &
    wait $!
  done
fi

container="${1:-}"
out="${2:-}"
if [ -z "$container" ] || [ -z "$out" ]; then
  echo "usage: $0 <container> <out-dir> --log-glob GLOB [--port N] [--container-port 8000] [--platform-prefix P] [--prom-file F] [--maintenance-file F]" >&2
  exit 64
fi
shift 2
glob=""
port=""
cport=8000
platform=""
promfile=""
maintfile=""
fixedport=0
while [ $# -gt 0 ]; do
  case "$1" in
    --log-glob) glob="$2"; shift ;;
    --port) port="$2"; fixedport=1; shift ;;
    --container-port) cport="$2"; shift ;;
    --platform-prefix) platform="$2"; shift ;;
    --prom-file) promfile="$2"; shift ;;
    --maintenance-file) maintfile="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done
if [ -z "$glob" ]; then
  echo "--log-glob is required: the engine log's path glob inside the container" >&2
  exit 64
fi

run_py() {
  if [ -n "${PROD_TELEMETRY_B64:-}" ]; then
    "$pybin" -c "$(printf '%s' "$PROD_TELEMETRY_B64" | base64 -d)" "$@"
  else
    "$pybin" "${PROD_TELEMETRY_PY:-$(dirname "$0")/prod_telemetry.py}" "$@"
  fi
}

mkdir -p "$out"
pidfile="$out/capture.pid"
followpid="$out/follower.pid"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then
  echo "a capture into $out is already running (pid $(cat "$pidfile")): not starting a second" >&2
  exit 75
fi
echo $$ > "$pidfile"

if [ -z "$port" ]; then
  mapped=$(docker port "$container" "${cport}/tcp" 2>/dev/null | head -1 | sed 's/.*://')
  port="${mapped:-$cport}"
fi

pypid=""
stop_all() {
  if [ -f "$followpid" ]; then
    kill "$(cat "$followpid" 2>/dev/null)" 2>/dev/null
  fi
  if [ -n "$pypid" ]; then
    sleep 1
    kill "$pypid" 2>/dev/null
    wait "$pypid" 2>/dev/null
  fi
  rm -f "$followpid"
  [ "$(cat "$pidfile" 2>/dev/null)" = "$$" ] && rm -f "$pidfile"
}
trap 'stop_all; exit 143' TERM INT HUP

urlfile="$out/metrics.url"
rm -f "$urlfile"
if [ "$fixedport" -eq 1 ]; then followurl="-"; urlarg=""; else followurl="$urlfile"; urlarg="--metrics-url-file $urlfile"; fi
sh "$0" --follower "$container" "$glob" "$followpid" "$followurl" "$cport" | run_py capture --out "$out" --metrics-url "http://127.0.0.1:${port}/metrics"   $urlarg ${platform:+--platform-prefix "$platform"} ${promfile:+--prom-file "$promfile"} ${maintfile:+--maintenance-file "$maintfile"} - &
pypid=$!
wait "$pypid"
rc=$?
stop_all
exit "$rc"
