"""Which DMA kernel source a launch builder loads at the width this process serves at, and the compile-time
defines that source needs.

The GDN state, conv-window and commit DMA kernels carry the pair's page counts as literals (384 recurrent-state
pages, 160 conv pages, 640 = 4 x 160 conv tasks) and those files are hashed into evidence, so the four-card
port never edits them. Each has a sibling `<stem>_tp.cpp` that reads the counts from defines instead, and this
module is the one place that picks between them: at the pair a builder gets its own file and no defines (the
launch is byte for byte what it was), at four cards the sibling and the defines derived from tp_shapes.

Stdlib only, py 3.7.
"""

from pathlib import Path

import tp_shapes


def source(path, environ=None):
    """`path` (a kernel .cpp) at the pair, its `_tp` sibling at any other width."""
    path = Path(path)
    if tp_shapes.chip_count(environ) == tp_shapes.PAIR:
        return str(path)
    return str(path.with_name(path.stem + '_tp' + path.suffix))


def defines(environ=None):
    """The (name, value) defines the `_tp` siblings read; empty at the pair, whose kernels take no defines."""
    if tp_shapes.chip_count(environ) == tp_shapes.PAIR:
        return []
    found = tp_shapes.active(environ)
    return [('QWEN_STATE_PAGES', str(found.gdn_state_pages)),
            ('QWEN_CONV_PAGES', str(found.gdn_conv_pages)),
            ('QWEN_CONV_TASKS', str(4 * found.gdn_conv_pages))]


def fold_defines(environ=None):
    """The define the attention fold and mask siblings read: the folded query rows per token (12 at the pair, 6 at four
    cards: one row per local query head). Empty at the pair, whose kernels take no defines."""
    if tp_shapes.chip_count(environ) == tp_shapes.PAIR:
        return []
    return [('QWEN_FOLD_HEAD_ROWS', str(tp_shapes.active(environ).attn_fold_rows))]
