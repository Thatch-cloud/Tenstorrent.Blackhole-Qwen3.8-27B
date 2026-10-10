# The card steps' telemetry sidecar (C2_TELEMETRY=1 in .github/c2-serving-job.env; qwen-c2-serving.yml sources this in the smoke and gate steps).
#
# It starts scripts/ci/card_telemetry.py in the background for the whole step: one read-only sample of every chip's ARC telemetry table per interval (AICLK and the
# limiter holding it, VCORE, power, current, temperatures, kernel NOP counters; the module docstring has the field map and why no ARC message is ever sent), a CSV and a
# summary in the results, one line per chip in the step's log. It is optional: whatever goes wrong here is a warning in the log, never a failed step.
#
# telemetry_start NAME OUT_DIR NODE...   unless TELEMETRY=1 this does nothing. Finds an interpreter that imports pyluwen (the host tt-smi's own, whose shebang names it;
#                                        else python3), reads every NODE (/dev/tenstorrent/<n>) once as a preflight, then starts the sampler (interval TELEMETRY_MS,
#                                        default 1000) that stops by itself when this shell exits. Start it AFTER the step's holder check and before the first container.
# telemetry_stop                         stops the sampler it started (SIGTERM, 20 s, then SIGKILL of that one pid), summarises its directory and gzips the csv. Safe to call twice.
# telemetry_summary DIR                  the summary of one directory (one line per chip on stdout).
# telemetry_summaries RESULTS            every RESULTS/telemetry-* directory that has no summary yet: the always-run step after a failed or cancelled job.
#
# Every function returns 0.
card_telemetry_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TELEMETRY_PID=
TELEMETRY_OUT=

# Candidates for the host's tt-smi, in order: TELEMETRY_SMI, the PATH, the runner user's own ~/.local/bin (HOME is a temp directory under Actions, so the account's
# home is asked of the passwd database; TELEMETRY_HOME overrides it).
telemetry_smi_candidates() {
  local home=${TELEMETRY_HOME:-}
  [ -z "${TELEMETRY_SMI:-}" ] || echo "$TELEMETRY_SMI"
  command -v tt-smi 2>/dev/null || true
  [ -n "$home" ] || home=$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || true)
  [ -z "$home" ] || echo "$home/.local/bin/tt-smi"
}

# The interpreter that can import pyluwen: the one named by a tt-smi script's shebang (its venv has pyluwen), else python3. Prints it; 1 when none can.
telemetry_python() {
  local smi line words py candidates=()
  while IFS= read -r smi; do
    [ -r "$smi" ] || continue
    line=
    IFS= read -r line < "$smi" || true
    case $line in
      '#!'*)
        read -r -a words <<< "${line#'#!'}"
        py=${words[0]:-}
        if [ "${py##*/}" = env ]; then py=${words[1]:-}; fi
        [ -z "$py" ] || candidates+=("$py")
        ;;
    esac
  done < <(telemetry_smi_candidates)
  candidates+=(python3)
  for py in "${candidates[@]}"; do
    if "$py" -c 'import pyluwen' > /dev/null 2>&1; then echo "$py"; return 0; fi
  done
  return 1
}

telemetry_unavailable() {  # name out reason
  echo "[TELEMETRY] $1: UNAVAILABLE ($3); the step runs without telemetry" >&2
  echo "::warning::C2_TELEMETRY=1 but $3"
  mkdir -p "$2"
  echo "[TELEMETRY] $1: UNAVAILABLE ($3)" > "$2/telemetry-summary.txt"
}

telemetry_start() {  # name out_dir node...
  [ "${TELEMETRY:-}" = 1 ] || return 0
  local name=$1 out=$2 py interval ms=${TELEMETRY_MS:-1000}
  shift 2
  if [ -n "${TELEMETRY_PID:-}" ]; then
    echo "[TELEMETRY] a sampler (pid $TELEMETRY_PID) is already running; $name not started" >&2
    return 0
  fi
  if [ "$#" = 0 ]; then telemetry_unavailable "$name" "$out" 'no card node was given'; return 0; fi
  if ! py=$(telemetry_python); then
    telemetry_unavailable "$name" "$out" 'no interpreter with pyluwen was found (the host tt-smi is expected to have one)'
    return 0
  fi
  mkdir -p "$out"
  # One synchronous read of every chip, so a decode that disagrees with pyluwen or a chip that cannot be read shows before any card time is spent.
  if ! timeout -k 5 90 "$py" -s "$card_telemetry_dir/card_telemetry.py" check --nodes "$@" > "$out/preflight.log" 2>&1; then
    cat "$out/preflight.log" >&2 || true
    telemetry_unavailable "$name" "$out" 'the preflight read of the chips failed'
    return 0
  fi
  sed 's/^/  /' "$out/preflight.log" | grep -E 'source|MISMATCH' || true
  interval=$(awk -v ms="$ms" 'BEGIN { printf "%.3f", ms / 1000 }')
  nice -n 10 "$py" -s "$card_telemetry_dir/card_telemetry.py" sample --out "$out" --nodes "$@" --interval "$interval" --parent "$$" > "$out/sampler.log" 2>&1 &
  TELEMETRY_PID=$!
  TELEMETRY_OUT=$out
  echo "[TELEMETRY] $name: sampling $# chip(s) every ${interval} s with $py (pid $TELEMETRY_PID) into ${out##*/}"
}

telemetry_summary() {  # dir
  [ -d "${1:-}" ] || return 0
  python3 -s "$card_telemetry_dir/card_telemetry.py" summary "$1" || true
}

telemetry_stop() {
  local pid=${TELEMETRY_PID:-} out=${TELEMETRY_OUT:-}
  [ -n "$pid" ] || return 0
  TELEMETRY_PID=
  TELEMETRY_OUT=
  kill -TERM "$pid" 2> /dev/null || true
  for _ in $(seq 1 40); do
    kill -0 "$pid" 2> /dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2> /dev/null; then
    echo "[TELEMETRY] the sampler (pid $pid) did not stop in 20 s; killing that pid" >&2
    kill -KILL "$pid" 2> /dev/null || true
    sleep 1
  fi
  if kill -0 "$pid" 2> /dev/null; then
    echo "[TELEMETRY] the sampler (pid $pid) is still alive after SIGKILL; leaving it (it stops when this shell exits)" >&2
  else
    wait "$pid" 2> /dev/null || true
  fi
  telemetry_summary "$out"
  return 0
}

telemetry_summaries() {  # results
  local dir
  for dir in "${1:-}"/telemetry-*/; do
    [ -d "$dir" ] || continue
    [ ! -f "$dir/telemetry-summary.json" ] || continue
    telemetry_summary "${dir%/}"
  done
  return 0
}
