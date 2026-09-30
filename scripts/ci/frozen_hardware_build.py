"""Direct, uncached hardware build of the frozen-recipe FP32 factory - the
65536 staging's own equivalent of scripts/ci/dspark_ladder_build.py's direct
call to dspark_fp32_build.main(), reused here instead of frozen_sim_build_cache.py's
caching wrapper.

Deliberately bypasses frozen_sim_build_cache.py: its content-addressed cache
key does not carry the backend (simulator vs hardware), so a prior
simulator-mode cache hit could silently short-circuit a hardware build (or a
hardware-mode entry could poison a later simulator run) - see
frozen_wide_chunk_normalization.py's module docstring. The ladder's own
hardware lane (scripts/ci/ladder-hardware-suite.sh) already establishes the
pattern of a direct, uncached hardware build; this mirrors it for the frozen
recipe's own staged modules (frozen_wide_chunk_scratch.factory_scope(),
dspark_fp32_build.main(hardware=True)) instead of the ladder's.

Only meaningful against a 65536-staged scripts/ci tree: frozen_wide_chunk_scratch.py
does not exist for any other context (frozen_wide_chunk_normalization.
adapt_wide_chunk_normalization() is a no-op below 65536; see that module's
docstring), and dspark_fp32_build.main() only accepts hardware=True at all
after frozen_wide_chunk_normalization._patch_fp32_build()'s graft.
"""

from frozen_wide_chunk_scratch import factory_scope
import dspark_fp32_build as baseline


def main():
    with factory_scope():
        baseline.main(hardware=True)


if __name__ == '__main__':
    main()
