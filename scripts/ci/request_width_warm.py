"""Compile and create the any-request engine's rows 1/2/4 state BEFORE the packed blocks capture (QWEN_FAST_M3_REQUEST_WARM).

The M3 attach builds one or two 64-row blocks. The first request engine built after the last capture compiles its rows 1/2/4
programs and creates its per-width state in the holes the most recently captured block freed, and that block's replays overwrite
whatever lives there: the next eager rows=2 warm-up of an engine build then hangs at its fence (the S8-1 and S8-3a hangs at eight
seats; the repo's own #48536 sequence). This module runs, once, at attach and before any trace is captured (one block:
serving_runtime.attach_combined_runtime, after the prefill warm; two blocks: between the phases of complete_blocks_two_phase),
exactly what an engine build runs eagerly at each width - the verify fixture, the forward under the TARGET_TAPS feature capture,
the pinned sampler's native-row sample, and for rows > 1 every publication prefix - so those programs exist and that state has
been allocated and released before any trace is captured. It captures no trace and keeps nothing: every tensor it allocates is
released, and the GDN state is saved before and restored after.

WIDTHS is (1, 2, 4). EVEN_WIDTHS is (1, 2, 4, 1), a gate-only discriminator: each eager forward flips the model CCL index and each
sample flips the sampler index, so three widths flip both parities for every later trace; four widths restore them.

The ModelBatch it builds is VerifierEngine.fixture's for the any-request engine serving_request_factory builds (norm_batch=True,
attention_replay=False, replay_group_rows=4, no short context, commit-only GDN beyond one row); request_fixture_options is that
keyword set, and test_request_width_warm holds it against what VerifierEngine.fixture records.
"""

from contextlib import ExitStack
import time

import force_argmax
import gdn_commit_dma
import gdn_multitoken_conv
from dflash_device import pindiag
from model_batch import ModelBatch

WIDTHS = (1, 2, 4)
EVEN_WIDTHS = (1, 2, 4, 1)
POSITION = 4096
MARKER = '[PINDIAG] request widths warmed before the packed traces'


def request_fixture_options(rows, retain):
    """The keyword arguments VerifierEngine.fixture passes ModelBatch for the any-request engine (norm_batch=True,
    attention_replay off, replay_group_rows=4, commit_only_gdn=True), without the pooled storage the engine's own build may add."""
    return dict(serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True, hoist_row_layout=True,
                device_loop_gdn=True, compact_prologue=True, batch_conv=True, packed_checkpoints=True,
                retain_records=retain, ordered_cache=True, norm_batch=True, attention_replay=False,
                attention_mask_once=False, replay_group_rows=4, short_context=False, attention_audit=False,
                **(dict(commit_only_gdn=True) if rows > 1 else {}))


def warm_request_widths(operations, model, helpers, sampler, page_width, widths=WIDTHS, position=POSITION, log=pindiag):
    """Run the request engine's eager warm-up at every width, in order, then release everything it allocated.

    `page_width` is the plugin's page-table width; the table is torch.arange(page_width), as prefill_warm_before_traces' is:
    at attach no request owns any page, and the table's last page must hold position + rows."""
    import torch
    import serving_runtime
    from dflash_request_runtime import TARGET_TAPS
    from prepared_target_features import PreparedTargetFeatures

    widths = tuple(widths)
    if (not widths or any(type(rows) is not int or rows not in force_argmax.SAMPLE_WIDTHS for rows in widths)
            or type(page_width) is not int or page_width <= 0 or type(position) is not int or position < 0
            or position + max(widths) > page_width * 64):
        raise ValueError('Supported request widths at a position inside the page table required')
    mesh = model.mesh_device
    pages = torch.arange(page_width, dtype=torch.int32).reshape(1, page_width)
    began, programs_before = time.perf_counter(), serving_runtime.program_count(model)
    with ExitStack() as scope:
        # The engine's `initial` snapshot: the live GDN state saved here and put back after every width (the engine's
        # restore_initial), so the warm-up leaves the live state as it found it.
        initial = [helper.allocate() for helper in helpers]
        scope.callback(lambda: gdn_multitoken_conv.release_owned(operations, [v for snapshot in initial for v in snapshot]))
        for helper, snapshot in zip(helpers, initial, strict=True):
            helper.save(snapshot)
        for rows in widths:
            for helper, snapshot in zip(helpers, initial, strict=True):
                helper.restore(snapshot)
            with ExitStack() as width:
                checkpoints = [helper.allocate() for helper in helpers]
                width.callback(lambda checkpoints=checkpoints: gdn_multitoken_conv.release_owned(
                    operations, [v for snapshot in checkpoints for v in snapshot]))
                features = []
                for tap in TARGET_TAPS:
                    features.append(operations.from_torch(torch.zeros((1, 1, rows, 5120), dtype=torch.bfloat16),
                        device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                        memory_config=operations.DRAM_MEMORY_CONFIG,
                        mesh_mapper=operations.ShardTensorToMesh(mesh, dim=3)))
                width.callback(lambda features=features: gdn_multitoken_conv.release_owned(operations, features))
                capture = PreparedTargetFeatures(model, TARGET_TAPS, features, copy=operations.copy,
                    storage_ids=lambda value: tuple(enumerate(gdn_multitoken_conv.addresses(operations, value))))
                width.callback(capture.close)
                fixture = ModelBatch(model, [1] * rows, position, pages, helpers, checkpoints, 0 if rows == 1 else rows,
                                     **request_fixture_options(rows, rows > 1))
                width.callback(fixture.close)
                logits = ids = None
                try:
                    with capture.capture():
                        logits = fixture.run(sharded_logits=True)
                    ids = force_argmax.sample_rows(sampler, logits, rows, operations, native_rows=True)
                    operations.synchronize_device(mesh)
                finally:
                    owned = [value for value in (logits, ids) if value is not None]
                    if owned:
                        gdn_multitoken_conv.release_owned(operations, owned)
                if rows > 1:
                    layers = [[*state.entry, result['states'], *result['packed_conv_states'], state.gdn.rec_state,
                               *state.gdn.conv_states, *checkpoint]
                              for state, result, checkpoint in fixture.retained.records]
                    publications = [gdn_commit_dma.prepare(mesh, layers, prefix) for prefix in range(rows + 1)]
                    for publication in publications:
                        publication()
                    operations.synchronize_device(mesh)
        for helper, snapshot in zip(helpers, initial, strict=True):
            helper.restore(snapshot)
        operations.synchronize_device(mesh)
    log(MARKER + ': rows={} programs={}->{} ms={:.0f}', widths, programs_before, serving_runtime.program_count(model),
        (time.perf_counter() - began) * 1000.0)
