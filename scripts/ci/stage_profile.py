"""Nested synchronized wall-time attribution, not a traced critical-path profiler."""

import ast
import inspect
import textwrap
import time
from contextlib import contextmanager


def direct_calls(source):
    tree = ast.parse(textwrap.dedent(source))
    return sorted({node.func.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"})


class StageProfile:
    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.enabled = False
        self.stack = []
        self.records = {}

    @contextmanager
    def scope(self, category, layer=None):
        if not self.enabled:
            yield
            return
        self.synchronize()
        if layer is None and self.stack:
            layer = self.stack[-1]["layer"]
        frame = dict(layer=layer, children=0.0, start=self.clock())
        self.stack.append(frame)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = self.clock() - frame["start"]
            self.stack.pop()
            if self.stack:
                self.stack[-1]["children"] += elapsed
            key = (category, layer)
            record = self.records.setdefault(key, dict(category=category, layer=layer, calls=0,
                                                        inclusive_ms=0.0, exclusive_ms=0.0))
            record["calls"] += 1
            record["inclusive_ms"] += elapsed * 1000
            record["exclusive_ms"] += (elapsed - frame["children"]) * 1000

    def wrap(self, category, function, layer=None):
        def measured(*args, **kwargs):
            with self.scope(category, layer):
                return function(*args, **kwargs)
        return measured

    def begin(self):
        if self.stack:
            raise RuntimeError("Cannot reset an active profiling scope")
        self.records.clear()
        self.enabled = True

    def finish(self):
        if self.stack:
            raise RuntimeError("Cannot finish an active profiling scope")
        self.enabled = False
        return list(self.records.values())


def decoder_bindings(layer, index, profiler):
    bindings = [(layer, "forward", profiler.wrap("decoder.block", layer.forward, index))]
    for name in direct_calls(inspect.getsource(type(layer).forward)):
        function = getattr(layer, name)
        if not callable(function) or name == "forward":
            raise ValueError(f"Unexpected native direct decoder call: {name}")
        bindings.append((layer, name, profiler.wrap(f"decoder.{name}", function, index)))
    return bindings
