"""Runtime-arg counts of the vglue row-mover launches against the program cache (hardware window N1, 2026-10-02).

generic_op caches a program on the kernel, defines, compile-time args and cores. It does NOT hash the length of a core's
runtime args or the io tensors, and a cache hit writes this launch's lists into the cached slots. N1: the window stack
(640 tasks, 49 words on a core) was cached first, then the unstack (1800 tasks, 129 / 137 words) hit the same entry and wrote
past the slots: 36,480 'Index N is larger than runtime args size 49' and a dispatch segfault. The fix pads every core's list
to one length and makes that length (the capacity) the last compile-time arg, so two launches that share a key always carry
identical per-core lengths. This module checks, on the CPU:

  * the declared reads: what the kernel reads (parsed from its source) fits the list the host builds, for 1-4 users;
  * the cache model: replaying a layer's launches against a cache keyed as generic_op keys them never hits an entry whose
    per-core lengths differ, in any user-count order, for the GDN row mover and the V3a attention fold.

Run at py 3.11: `py -3.11 -m unittest test_tp4_vglue_argcount` from scripts/ci.
"""

import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import attention_block_fold_tp as fold
import gdn_block_conv_tp as block
import gdn_rows_dma_tp as rows_dma
import tp_shapes
from test_tp4_vglue_attention import chunks_for

HERE = Path(__file__).resolve().parent
CORES = 110
WIDTH = 4120                    # the TP4 packed projection
SEGMENTS = (((0, 16), (16, 32), (32, 48), (48, 64)), ((0, 16),), ((0, 32), (32, 48)), ((0, 32),))
ADDRESSES = list(range(40))


def four():
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def kernel_constant(name, text):
    return int(re.search(r'constexpr uint32_t %s = (\d+);' % name, text).group(1))


class Cache(object):
    """generic_op's program cache as N1 showed it: no runtime-arg lengths, no io tensors in the key."""

    def __init__(self):
        self.entries = {}

    def launch(self, key, lengths):
        """The cached per-core lengths on a hit (None on a miss, which caches this launch's)."""
        if key in self.entries:
            return self.entries[key]
        self.entries[key] = lengths
        return None


def key_of(module, per_core, source, destination, defines, cb):
    capacity = getattr(module, 'capacity', None)
    compile_args = (source, destination) + ((capacity(per_core),) if capacity else ())
    return (module.KERNEL, defines, compile_args, tuple(range(len(per_core))), cb)


class DeclaredReadTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def plans(self, users):
        found = tp_shapes.active()
        return {
            'split': rows_dma.split_pieces(users, WIDTH),
            'canon': rows_dma.canon_block(users, WIDTH),
            'merge': rows_dma.merge_outputs(users, found.gdn_a_col - found.gdn_qkv),
            'windows': block.plan_windows(users),
            'unstack': block.plan_unstack(users, WIDTH),
        }

    def test_the_gdn_kernel_reads_what_the_host_declares(self):
        text = (HERE / 'gdn_rows_dma_tp.cpp').read_text()
        self.assertEqual(kernel_constant('TASK_WORDS', text), rows_dma.TASK_WORDS)
        # the highest word a task reads is base + 2 + 1 * 3 + 2 = base + 7: inside TASK_WORDS
        self.assertIn('base + 2 + half * 3 + 2', text)
        self.assertEqual(rows_dma.TASK_WORDS, 2 + 3 + 3)
        self.assertIn('get_compile_time_arg_val(destination_args.next_compile_time_args_offset())', text)
        for users in (1, 2, 3, 4):
            for name, tasks in self.plans(users).items():
                per_core = rows_dma.distribute(tasks, CORES)
                arguments = rows_dma.runtime_arguments(per_core, ADDRESSES, ADDRESSES)
                size = 1 + rows_dma.TASK_WORDS * rows_dma.capacity(per_core)
                for core, words in zip(per_core, arguments):
                    self.assertEqual(len(words), size, (users, name))
                    self.assertEqual(words[0], len(core))
                    self.assertTrue(all(word == 0 for word in words[1 + rows_dma.TASK_WORDS * len(core):]))

    def test_the_capacity_is_the_last_compile_time_arg_of_the_launch(self):
        for module in (rows_dma, fold):
            source = (HERE / (module.__name__ + '.py')).read_text()
            self.assertIn('compile_time_args=[*source_layouts[0], *destination_layouts[0], capacity(per_core)]', source)

    def test_the_fold_kernel_reads_what_the_host_declares(self):
        text = (HERE / 'attention_block_fold_tp.cpp').read_text()
        self.assertIn('base = 2 + index * %d;' % fold.TASK_WORDS, text)
        self.assertIn('get_compile_time_arg_val(destination_args.next_compile_time_args_offset())', text)
        for segments in SEGMENTS:
            chunks = chunks_for(segments)
            for inverse, plan in ((False, fold.forward_tasks(chunks)), (True, fold.inverse_tasks(chunks))):
                flat = fold.flat_tasks(plan, [1] * len(chunks), [2] * len(chunks))
                per_core = fold.distribute(flat, CORES)
                size = 2 + fold.TASK_WORDS * fold.capacity(per_core)
                for core, words in zip(per_core, fold.runtime_arguments(inverse, per_core)):
                    self.assertEqual(len(words), size)
                    self.assertEqual(words[1], len(core))


class CacheModelTests(unittest.TestCase):
    def setUp(self):
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)

    def gdn_layer(self, users):
        """One layer's launches in order: (tasks, source placement, destination placement)."""
        found = tp_shapes.active()
        return [
            (rows_dma.split_pieces(users, WIDTH), 'L1', 'L1'),
            (rows_dma.canon_block(users, WIDTH), 'L1', 'DRAM'),
            (block.plan_windows(users), 'DRAM', 'DRAM'),
            (block.plan_unstack(users, WIDTH), 'DRAM', 'DRAM'),
            (rows_dma.merge_outputs(users, found.gdn_a_col - found.gdn_qkv), 'DRAM', 'DRAM'),
        ]

    def test_no_gdn_launch_hits_a_cache_entry_of_another_length(self):
        for order in ((4, 4), (2, 3, 4), (4, 3, 2), (3, 2, 4, 1), (1, 2, 3, 4, 4, 3, 2, 1)):
            cache = Cache()
            for users in order:
                for tasks, source, destination in self.gdn_layer(users):
                    per_core = rows_dma.distribute(tasks, min(len(tasks), CORES))
                    lengths = tuple(len(words) for words in rows_dma.runtime_arguments(per_core, ADDRESSES, ADDRESSES))
                    hit = cache.launch(key_of(rows_dma, per_core, source, destination, (('CANON_DENORM', '1'),), 16384), lengths)
                    if hit is not None:
                        self.assertEqual(hit, lengths, 'order %r users %d: a hit would write %d words into %d word slots' % (
                            order, users, max(lengths), max(hit)))

    def test_the_n1_pair_is_the_window_stack_then_the_unstack(self):
        self.assertEqual((len(block.plan_windows(4)), len(block.plan_unstack(4, WIDTH))), (640, 1800))

    def test_fold_in_and_fold_out_do_not_hit_an_entry_of_another_length(self):
        for segments in SEGMENTS:
            chunks = chunks_for(segments)
            cache = Cache()
            for _ in range(2):
                for inverse, plan in ((False, fold.forward_tasks(chunks)), (True, fold.inverse_tasks(chunks))):
                    flat = fold.flat_tasks(plan, [1] * len(chunks), [2] * len(chunks))
                    per_core = fold.distribute(flat, CORES)
                    lengths = tuple(len(words) for words in fold.runtime_arguments(inverse, per_core))
                    key = key_of(fold, per_core, 'DRAM', 'DRAM', (('QWEN_FOLD_HEAD_ROWS', '6'),), 18432)
                    hit = cache.launch(key, lengths)
                    if hit is not None:
                        self.assertEqual(hit, lengths, 'segments %r' % (segments,))


if __name__ == '__main__':
    unittest.main()
