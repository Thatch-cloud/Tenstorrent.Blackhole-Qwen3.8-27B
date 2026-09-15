"""Run the unchanged matched-context checks against the qualified maxima build."""

from pathlib import Path
import runpy

from dspark_splitk_maxima_hardware import hardware_identity


if __name__ == '__main__':
    with hardware_identity():
        runpy.run_path(str(Path(__file__).with_name('matched-context-attention-probe.py')), run_name='__main__')
