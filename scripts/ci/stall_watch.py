"""Stall watch: name the stuck work before anything touches the device (QWEN_FAST_STALL_DEADLINE_S, off by default).

The audits-off four-user hang never named what was stuck: the process sat below Python, the watchdog that existed
(trace_census.watched_step) covered sequential steps only and ended the process with faulthandler's exit=True, which killed the
evidence with it. This module extends the watch to engine builds, prefill segments and packed rounds, and changes what happens
on a stall: the stacks are dumped, the pinned tt-metal triage tools read the hung mesh, and only then is the process ended.

    scope(kind, label)   a context manager around one stretch of serving work. kind is 'step' (a sequential step or a packed
                         round), 'build' (an engine build) or 'prefill' (one prefill segment); each has its own deadline.
                         Nested scopes are no-ops (the outermost one covers them). Off, it is a nullcontext.

On a stall, all of these, in this order, none of them touching the device through the serving process:
  1. faulthandler.dump_traceback_later(deadline, exit=False): a C thread, so it fires while the stalled call holds the GIL. It
     writes every host thread's Python stack to stderr and does NOT end the process.
  2. a sidecar PROCESS (python stall_watch.py --sidecar <pid>), started at the first scope, told by a pipe when a scope is armed
     and disarmed. It is a separate process on purpose: a stalled call that holds the GIL starves any thread of this one, and
     the triage must run regardless. When the armed deadline passes it logs '[STALL] ...' and runs the triage tools
     (dump_running_operations, dump_callstacks, check_binary_integrity, check_noc_status, dump_fast_dispatch, check_eth_status
     from the pinned tt-metal's tools/triage) one after another with a timeout each, their output logged as '[TRIAGE <tool>]'.
  3. only after the triage has finished (or timed out) does the sidecar end the serving process (SIGKILL), unless
     QWEN_FAST_STALL_KILL=0, which leaves it hung for a person to look at. The cards are never reset from here.
So the device is untouched until the triage has read it; the container is killed after, and the next job's card reset follows.

Flags (all optional; the watch is on only with QWEN_FAST_STALL_DEADLINE_S):
  QWEN_FAST_STALL_DEADLINE_S   seconds for a 'step' (a packed round is ~0.17 s, a sequential step under a second)
  QWEN_FAST_STALL_BUILD_S      seconds for a 'build', default 600 (an engine build compiles hundreds of programs and captures)
  QWEN_FAST_STALL_PREFILL_S    seconds for a 'prefill', default 900 (a 60000-token segment)
  QWEN_FAST_TRIAGE_ROOT        the triage tools' directory, default /opt/tt-metal/tools/triage
  QWEN_FAST_TRIAGE_TOOLS       comma list overriding the six tools
  QWEN_FAST_TRIAGE_TIMEOUT_S   per tool, default 180
  QWEN_FAST_TRIAGE_CMD         a command template overriding the two tried by default ('{python} {root}/{tool}.py', then
                               '{python} {root}/triage.py --run={tool}'): the first that exits 0 is the tool's reading
  QWEN_FAST_STALL_KILL         '0' keeps the hung process alive after the triage

UNVERIFIED on hardware: that the triage tools run from inside the container against a live hung mesh, and with which command line.
Stdlib only.
"""

import faulthandler
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext

DEADLINE_FLAG = 'QWEN_FAST_STALL_DEADLINE_S'
BUILD_FLAG = 'QWEN_FAST_STALL_BUILD_S'
PREFILL_FLAG = 'QWEN_FAST_STALL_PREFILL_S'
ROOT_FLAG = 'QWEN_FAST_TRIAGE_ROOT'
TOOLS_FLAG = 'QWEN_FAST_TRIAGE_TOOLS'
TIMEOUT_FLAG = 'QWEN_FAST_TRIAGE_TIMEOUT_S'
COMMAND_FLAG = 'QWEN_FAST_TRIAGE_CMD'
KILL_FLAG = 'QWEN_FAST_STALL_KILL'
DEFAULT_ROOT = '/opt/tt-metal/tools/triage'
DEFAULT_BUILD_S = 600.0
DEFAULT_PREFILL_S = 900.0
DEFAULT_TIMEOUT_S = 180.0
TOOLS = ('dump_running_operations', 'dump_callstacks', 'check_binary_integrity', 'check_noc_status', 'dump_fast_dispatch',
         'check_eth_status')
COMMANDS = ('{python} {root}/{tool}.py', '{python} {root}/triage.py --run={tool}')
KINDS = ('step', 'build', 'prefill')
OUTPUT_LINES = 160
LINE_WIDTH = 240


def _seconds(text, name):
    try:
        seconds = float(text)
    except ValueError:
        seconds = 0.0
    if not 0 < seconds < float('inf'):
        raise ValueError('%s must be a positive number of seconds, got %r' % (name, text))
    return seconds


def deadline(kind, environ=None):
    """The deadline in seconds for a scope of this kind, or None when the watch is off (QWEN_FAST_STALL_DEADLINE_S unset).
    A bad value is a configuration error."""
    environ = os.environ if environ is None else environ
    if kind not in KINDS:
        raise ValueError('scope kind must be one of %s, got %r' % (', '.join(KINDS), kind))
    base = environ.get(DEADLINE_FLAG)
    if base is None or base == '':
        return None
    step = _seconds(base, DEADLINE_FLAG)
    if kind == 'step':
        return step
    name, default = (BUILD_FLAG, DEFAULT_BUILD_S) if kind == 'build' else (PREFILL_FLAG, DEFAULT_PREFILL_S)
    text = environ.get(name)
    return default if text is None or text == '' else _seconds(text, name)


def enabled(environ=None):
    return deadline('step', environ) is not None


def triage_tools(environ=None):
    environ = os.environ if environ is None else environ
    text = environ.get(TOOLS_FLAG)
    if not text:
        return TOOLS
    return tuple(name.strip() for name in text.split(',') if name.strip())


def triage_commands(tool, environ=None, python=None):
    """The argv lists to try, in order, for one triage tool."""
    environ = os.environ if environ is None else environ
    templates = (environ[COMMAND_FLAG],) if environ.get(COMMAND_FLAG) else COMMANDS
    values = dict(python=python or sys.executable or 'python3', root=environ.get(ROOT_FLAG) or DEFAULT_ROOT, tool=tool)
    return [template.format(**values).split() for template in templates]


def triage_problems(environ=None, find_spec=None):
    """What stops the triage tools running in THIS interpreter: [] when the root, every tool and the ttexalens module they import
    are present, else one sentence per problem. Never raises."""
    environ = os.environ if environ is None else environ
    root = environ.get(ROOT_FLAG) or DEFAULT_ROOT
    problems = []
    try:
        if not os.path.isdir(root):
            problems.append('the triage directory %s does not exist' % root)
        else:
            absent = [tool for tool in triage_tools(environ)
                      if not (os.path.isfile(os.path.join(root, tool + '.py')) or os.path.isfile(os.path.join(root, 'triage.py')))]
            if absent:
                problems.append('missing from %s: %s' % (root, ', '.join(absent)))
        if find_spec is None:
            import importlib.util
            find_spec = importlib.util.find_spec
        if find_spec('ttexalens') is None:
            problems.append('the ttexalens module is not installed for %s (the image build installs the pinned '
                            '%s/requirements.txt)' % (sys.executable or 'python3', root))
    except BaseException as failure:
        problems.append('the triage readiness check failed: %s: %s' % (type(failure).__name__, str(failure)[:80]))
    return problems


def say(message):
    """One flushed stderr line (the serving log): a stalled process cannot be trusted with a logger."""
    for line in str(message).splitlines() or ['']:
        sys.stderr.write(line[:LINE_WIDTH] + '\n')
    sys.stderr.flush()


def run_tool(tool, environ=None, runner=subprocess.run, log=say, timeout=None):
    """One triage tool: each candidate command until one exits 0, its output logged as '[TRIAGE <tool>]' lines. -> True when
    a command ran to a zero exit. A command that is missing, times out or fails is logged and the next is tried."""
    environ = os.environ if environ is None else environ
    timeout = _seconds(environ.get(TIMEOUT_FLAG, DEFAULT_TIMEOUT_S), TIMEOUT_FLAG) if timeout is None else timeout
    for argv in triage_commands(tool, environ):
        log('[TRIAGE %s] begin %s' % (tool, ' '.join(argv)))
        try:
            done = runner(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, universal_newlines=True)
        except subprocess.TimeoutExpired as failure:
            output = failure.output
            log('[TRIAGE %s] timed out after %gs' % (tool, timeout))
            _emit(tool, output, log)
            continue
        except BaseException as failure:
            log('[TRIAGE %s] could not run: %s: %s' % (tool, type(failure).__name__, str(failure)[:120]))
            continue
        _emit(tool, done.stdout, log)
        log('[TRIAGE %s] exit %s' % (tool, done.returncode))
        if done.returncode == 0:
            return True
    return False


def _emit(tool, output, log):
    if isinstance(output, bytes):
        output = output.decode('utf-8', 'replace')
    lines = (output or '').splitlines()
    for line in lines[:OUTPUT_LINES]:
        log('[TRIAGE %s] %s' % (tool, line))
    if len(lines) > OUTPUT_LINES:
        log('[TRIAGE %s] ... %d more lines' % (tool, len(lines) - OUTPUT_LINES))


def run_triage(environ=None, runner=subprocess.run, log=say):
    """Every triage tool in order. -> {tool: bool}. Never raises."""
    results = {}
    for tool in triage_tools(environ):
        try:
            results[tool] = run_tool(tool, environ, runner, log)
        except BaseException as failure:
            log('[TRIAGE %s] failed: %s: %s' % (tool, type(failure).__name__, str(failure)[:120]))
            results[tool] = False
    return results


# --- the sidecar --------------------------------------------------------------------------------------------------------


def sidecar(lines, parent, environ=None, runner=subprocess.run, log=say, kill=None, clock=time.time):
    """The sidecar's loop over the parent's lines: 'arm <epoch deadline> <label>', 'disarm', 'quit'. End of input (the parent
    is gone) ends it quietly. When an armed deadline passes it logs the stall, runs the triage, and then ends the parent unless
    QWEN_FAST_STALL_KILL=0. -> 'quit' | 'eof' | 'stalled'."""
    environ = os.environ if environ is None else environ
    kill = kill or (lambda pid: os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM)))
    inbox = queue.Queue()

    def read():
        for line in lines:
            inbox.put(line.rstrip('\n'))
        inbox.put(None)

    threading.Thread(target=read, name='stall-sidecar-reader', daemon=True).start()
    armed = None
    while True:
        wait = None if armed is None else max(armed[0] - clock(), 0.0)
        try:
            item = inbox.get(timeout=wait)
        except queue.Empty:
            stalled(armed, parent, environ, runner, log, kill)
            return 'stalled'
        if item is None:
            return 'eof'
        word, _, rest = item.partition(' ')
        if word == 'quit':
            return 'quit'
        if word == 'disarm':
            armed = None
        elif word == 'arm':
            when, _, label = rest.partition(' ')
            try:
                armed = (float(when), label)
            except ValueError:
                log('[STALLWATCH] ignored a bad arm line: %s' % item[:80])


def stalled(armed, parent, environ, runner, log, kill):
    when, label = armed
    log('[STALL] %s did not finish by its deadline: the serving process is stalled in it. Stacks of every host thread follow from '
        'faulthandler (the process is NOT ended yet); then the tt-metal triage reads the hung mesh.' % label)
    problems = triage_problems(environ)
    if problems:
        log('[TRIAGE-CHECK] UNAVAILABLE, the device triage will not give a reading: %s' % '; '.join(problems))
    else:
        log('[TRIAGE-CHECK] ready: the triage tools and ttexalens are present')
    results = run_triage(environ, runner, log)
    log('[STALL] triage finished: %s' % ' '.join('%s=%s' % (tool, 'ok' if ok else 'failed') for tool, ok in results.items()))
    if environ.get(KILL_FLAG, '1') == '0':
        log('[STALL] QWEN_FAST_STALL_KILL=0: the serving process is left hung for inspection; reset the cards before reuse')
        return
    log('[STALL] ending the serving process (pid %s) now that the triage has read the device' % parent)
    try:
        kill(parent)
    except BaseException as failure:
        log('[STALL] could not end pid %s: %s: %s' % (parent, type(failure).__name__, str(failure)[:80]))


# --- the in-process side --------------------------------------------------------------------------------------------------


class Watch:
    """The serving process's end: starts the sidecar at the first scope, arms faulthandler and the sidecar around the outermost
    scope. One per process (WATCH)."""

    def __init__(self, spawn=None, clock=time.time):
        self.spawn = spawn or self.spawn_sidecar
        self.clock = clock
        self.lock = threading.Lock()
        self.depth = 0
        self.pipe = None
        self.process = None

    @staticmethod
    def spawn_sidecar():
        return subprocess.Popen([sys.executable or 'python3', '-u', os.path.abspath(__file__), '--sidecar', str(os.getpid())],
                                stdin=subprocess.PIPE, universal_newlines=True, bufsize=1, close_fds=True)

    def start(self):
        if self.pipe is not None:
            return
        try:
            self.process = self.spawn()
            self.pipe = self.process.stdin
        except BaseException as failure:
            self.pipe = False
            say('[STALL] sidecar unavailable (%s: %s): stacks only, no triage' % (type(failure).__name__, str(failure)[:80]))

    def send(self, line):
        if not self.pipe:
            return
        try:
            self.pipe.write(line + '\n')
            self.pipe.flush()
        except BaseException:
            self.pipe = False

    @contextmanager
    def scope(self, kind, label, environ=None):
        seconds = deadline(kind, environ)
        if seconds is None:
            yield
            return
        with self.lock:
            nested = self.depth > 0
            self.depth += 1
        if nested:
            try:
                yield
            finally:
                with self.lock:
                    self.depth -= 1
            return
        self.start()
        text = '%s %s' % (kind, label)
        self.send('arm %.3f %s' % (self.clock() + seconds, text.replace('\n', ' ')[:120]))
        try:
            faulthandler.dump_traceback_later(seconds, repeat=False, exit=False, file=sys.stderr)
        except BaseException as failure:
            say('[STALL] faulthandler unavailable (%s): the sidecar still triages' % type(failure).__name__)
        try:
            yield
        finally:
            try:
                faulthandler.cancel_dump_traceback_later()
            except BaseException:
                pass
            self.send('disarm')
            with self.lock:
                self.depth -= 1

    def close(self):
        self.send('quit')
        pipe, self.pipe = self.pipe, None
        if pipe:
            try:
                pipe.close()
            except BaseException:
                pass


WATCH = Watch()


def scope(kind, label):
    """The serving code's seam: a context manager around one stretch of work. A no-op without QWEN_FAST_STALL_DEADLINE_S."""
    if not enabled():
        return nullcontext()
    return WATCH.scope(kind, label)


def main(argv):
    if len(argv) == 3 and argv[1] == '--sidecar':
        sidecar(sys.stdin, int(argv[2]))
        return 0
    sys.stderr.write('usage: stall_watch.py --sidecar <pid>\n')
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
