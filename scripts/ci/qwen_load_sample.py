"""Sample the real hardware process without changing model or kernel behavior."""

import faulthandler
import json
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback


def snapshot(frame, elapsed, read_text):
    metrics = {}
    for name in ('/proc/self/io', '/proc/self/status', '/proc/pressure/io',
                 '/sys/fs/cgroup/cpu.stat', '/sys/fs/cgroup/memory.current',
                 '/sys/fs/cgroup/memory.events', '/sys/fs/cgroup/io.stat'):
        try:
            contents = read_text(name)
            if name.endswith('/status'):
                contents = '\n'.join(line for line in contents.splitlines()
                    if line.startswith(('VmRSS:', 'VmSwap:', 'Threads:')))
            metrics[name] = contents
        except OSError as error:
            metrics[name] = type(error).__name__
    stack = [dict(file=entry.filename, line=entry.lineno, function=entry.name)
        for entry in traceback.extract_stack(frame)]
    return dict(stage='load_sample', elapsed_seconds=elapsed, stack=stack,
        metrics=metrics, performance_qualified=False)


def main():
    entry = Path(sys.argv[1]).resolve(strict=True)
    if entry.parent != Path(__file__).resolve().parent:
        raise ValueError('Only sibling experiment entrypoints can be sampled')
    sys.argv = [str(entry), *sys.argv[2:]]
    stopped = threading.Event()
    main_thread = threading.get_ident()
    started = time.monotonic()

    def sample():
        while not stopped.wait(10):
            frame = sys._current_frames().get(main_thread)
            if frame is not None:
                record = snapshot(frame, time.monotonic() - started,
                    lambda name: Path(name).read_text())
                print(json.dumps(record), flush=True)

    worker = threading.Thread(target=sample, name='load-sampler', daemon=True)
    faulthandler.dump_traceback_later(30, repeat=True)
    worker.start()
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        stopped.set()
        worker.join(timeout=2)
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
