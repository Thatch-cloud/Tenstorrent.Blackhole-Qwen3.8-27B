# SPDX-License-Identifier: Apache-2.0
"""K3 — retain the post-final-norm hidden state for the MTP proposer.

Every decode path in `tt/model.py` ends:

    x = self._final_norm_decode(x)
    logits = self._lm_head(x)

so `_lm_head`'s input *is* the hidden state the MTP head consumes -- the same tensor vLLM
passes to the drafter. There are **13 `_lm_head` call sites** across prefill, decode, chunked
prefill and the TP variants, so hooking that one method covers all of them without touching a
single decode path.

Off unless `QWEN36_LOAD_MTP=1`, so the default path is unchanged.

The copy is into a **preallocated** buffer, not an attribute rebind: a rebind would swap the
tensor address mid-trace, which is the constraint `reset_state_inplace` and the K2 stager exist
to satisfy. The buffer is (re)allocated when the shape changes -- prefill and decode differ --
which happens outside trace capture; steady-state decode takes the copy path.

    python3 tt-patch-hidden-retention.py <path-to-model.py>
"""
import io
import sys

ANCHOR = '''    def _lm_head(self, x):
        """LM-head matmul. Vocab-sharded mesh: partial logits + all-gather to full replicated.
        Single device: plain matmul."""
'''

REPLACEMENT = '''    def _mtp_retain_enabled(self):
        """Read per call rather than caching, so a test can toggle it without rebuilding."""
        return os.environ.get("QWEN36_LOAD_MTP") == "1"

    def stash_hidden(self, x):
        """Retain the post-final-norm hidden state for the MTP proposer.

        Copy into a preallocated buffer rather than rebinding: a rebind changes the tensor
        address, which a captured decode trace cannot tolerate. Reallocate only when the shape
        changes (prefill vs decode), which happens outside trace capture.
        """
        buf = getattr(self, "_last_hidden", None)
        if buf is None or list(buf.shape) != list(x.shape):
            self._last_hidden = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return
        ttnn.copy(x, buf)

    def _lm_head(self, x):
        """LM-head matmul. Vocab-sharded mesh: partial logits + all-gather to full replicated.
        Single device: plain matmul."""
        # MTP hidden-state retention. _lm_head's input is exactly what the MTP head consumes,
        # and hooking here covers all 13 call sites without touching any decode path.
        if self._mtp_retain_enabled():
            self.stash_hidden(x)
'''


def main(path):
    src = io.open(path, encoding="utf-8").read()
    if "stash_hidden" in src:
        print(f"PATCH already applied: {path}")
        return 0
    if ANCHOR not in src:
        print("PATCH FAILED: _lm_head anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, REPLACEMENT, 1)
    io.open(path, "w", encoding="utf-8", newline="\n").write(src)
    print(f"PATCH applied: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
