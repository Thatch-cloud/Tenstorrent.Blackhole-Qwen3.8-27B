"""Loaded through PYTHONPATH by the interpreters the tests start (see hide_nvml.py)."""

try:
    import hide_nvml
    hide_nvml.install()
except Exception:      # a stub that fails to load must never stop an interpreter
    pass
