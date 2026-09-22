"""Bounded owned draft features at an absolute target prefill frontier."""

import os
from gdn_multitoken_conv import addresses
from contextlib import contextmanager
import operator
from model_batch import instance_overrides
from target_features import LayerOutputCapture

# The absolute prefill position (prompt frontier) the fast path admits. The
# serving default stays 65,504 - the target allocation and verification headroom
# the fleet is qualified for. QWEN_FAST_MAX_POSITION raises it for a measurement
# only (e.g. a single-user 163,840 acceptance run); it changes nothing unless set,
# following the QWEN_FAST_* / QWEN_DSPARK_* measurement-gate convention.
MAX_POSITION = int(os.environ.get('QWEN_FAST_MAX_POSITION', '65504'))


# Every batched prefill entry point whose third positional argument is empty_slots.
# prefill_paged_slots is the plugin's own; prefill_paged_slots_range is the resumable
# form M1 adds (lever_n_model_patch.py), whose signature is
# (token_ids_list, page_table, empty_slots, starts, ends, valid_lens=None) - the same
# first three, so one wrapper serves both. A name outside this set is refused at
# capture time rather than silently unwrapped.
BATCHED_PREFILL_ENTRIES = frozenset({'prefill_paged_slots', 'prefill_paged_slots_range'})


def prefill_slot(empty_slots):
    """The one native GDN slot a batched prefill writes.

    The lifecycle admits one fresh prompt per prefill, so the plugin's empty_slots
    names exactly one decode slot - [0] for the first user, [1] for the second. Any
    other count is a broken serving contract, not a batch to record.
    """
    try:
        slots = list(empty_slots)
    except TypeError:
        slots = None
    if slots is None or len(slots) != 1 or isinstance(slots[0], bool):
        raise ValueError('Fast serving prefills exactly one user per call; got slots %r' % (empty_slots,))
    try:
        slot = operator.index(slots[0])
    except TypeError:
        raise ValueError('Integer native GDN slot required; got %r' % (slots[0],)) from None
    if slot < 0:
        raise ValueError('Non-negative native GDN slot required; got %r' % (slot,))
    return slot


def prefill_window(position):
    if type(position) is not int or not 1 <= position <= MAX_POSITION:
        raise ValueError('Absolute prefill position within the target allocation and verification headroom required')
    return dict(start=max(0, position - 2048), end=position, rows=min(position, 2048))


def chunk_window(position, chunk_start, valid_rows):
    window = prefill_window(position)
    if (type(chunk_start) is not int or type(valid_rows) is not int or chunk_start < 0
            or valid_rows < 1 or chunk_start + valid_rows > position):
        raise ValueError('Explicit in-bounds absolute chunk coordinates required')
    start, end = max(window['start'], chunk_start), min(position, chunk_start + valid_rows)
    return dict(start=start, end=max(start, end), rows=max(0, end - start))


def validate_prefill_chunks(position, chunks):
    cursor, pieces = 0, []
    for chunk in chunks:
        if (chunk.get('chunk_start') != cursor or type(chunk.get('bucket')) is not int
                or not 32 <= chunk['bucket'] <= 2048 or chunk['bucket'] % 32
                or type(chunk.get('valid_rows')) is not int or not 1 <= chunk['valid_rows'] <= chunk['bucket']):
            raise ValueError('Complete ordered native prefill chunks and explicit valid lengths required')
        piece = chunk_window(position, chunk['chunk_start'], chunk['valid_rows'])
        if chunk.get('retained_rows') != piece['rows']:
            raise ValueError('Recorded retained rows must equal the valid tail intersection')
        if piece['rows']:
            pieces.append(piece)
        cursor += chunk['valid_rows']
    if cursor != position or sum(piece['rows'] for piece in pieces) != prefill_window(position)['rows']:
        raise ValueError('Prefill chunks must cover the full prompt and the exact draft tail')
    return pieces


def snapshot_prefill_tail(operations, value, position, *, checks=None, chunk_start=0, valid_rows=None):
    valid_rows = position if valid_rows is None else valid_rows
    window = chunk_window(position, chunk_start, valid_rows)
    if (not window['rows'] or len(value.shape) != 4 or tuple(value.shape)[:2] != (1, 1) or value.shape[2] < valid_rows
            or value.shape[3] != 2560 or value.dtype != operations.bfloat16):
        raise ValueError('Complete TP2 BF16 prefill feature output required; chunks cannot stand in for the full context')
    sliced = output = None
    try:
        sliced = operations.slice(value, (0, 0, window['start'] - chunk_start, 0),
            (1, 1, window['end'] - chunk_start, 2560))
        output = operations.clone(sliced, memory_config=operations.DRAM_MEMORY_CONFIG)
        if checks is not None:
            import torch

            originals = operations.get_device_tensors(value)
            snapshots = operations.get_device_tensors(output)
            if len(originals) != 2 or len(snapshots) != 2:
                raise AssertionError('Both prefill feature shards required')
            for chip, (original, snapshot) in enumerate(zip(originals, snapshots, strict=True)):
                expected = operations.to_torch(original)[..., window['start'] - chunk_start:window['end'] - chunk_start, :].contiguous()
                actual = operations.to_torch(snapshot).contiguous()
                if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                    raise AssertionError('Prefill tail snapshot changed, shifted or included padded feature rows')
                checks.append(dict(chip=chip, **window, exact=True))
        return output
    except BaseException:
        if output is not None:
            operations.deallocate(output)
        raise
    finally:
        if sliced is not None and addresses(operations, sliced) != addresses(operations, value):
            operations.deallocate(sliced)


class PrefillWindowCapture:
    def __init__(self, operations, model, position, taps, *, checks=None):
        self.operations, self.model, self.position, self.taps = operations, model, position, tuple(taps)
        self.window = prefill_window(position)
        if not callable(getattr(model, '_forward_prefill_chunk_masked_tp', None)):
            raise ValueError('Pinned native TP slot-prefill chunk boundary required')
        self.checks, self.chunks, self.children, self.merged = checks, [], [], []
        self.cursor = 0
        self.started = self.active = self.complete = self.closed = False
        self.result = None
        # The native GDN slot the batched prefill wrote this user's state into, or
        # None when the single-sequence prefill_traced_chunked path ran instead.
        self.prefill_slot = None

    def wrap(self, original):
        def chunk(token_buf, valid_len, chunk_start, page_table, bucket, *args, **kwargs):
            if not self.active or chunk_start != self.cursor:
                raise ValueError('Prefill chunks must execute once in absolute sequence order')
            piece = chunk_window(self.position, chunk_start, valid_len)
            if type(bucket) is not int or bucket % 32 or not valid_len <= bucket <= 2048:
                raise ValueError('Bounded native padded chunk geometry required')
            captured = None
            if piece['rows']:
                captured = LayerOutputCapture(self.model, self.taps,
                    snapshot=lambda value: snapshot_prefill_tail(self.operations, value, self.position,
                        checks=self.checks, chunk_start=chunk_start, valid_rows=valid_len),
                    release=self.operations.deallocate,
                    storage_ids=lambda value: tuple(enumerate(addresses(self.operations, value))))
                self.children.append(captured)
                with captured.capture():
                    output = original(token_buf, valid_len, chunk_start, page_table, bucket, *args, **kwargs)
            else:
                output = original(token_buf, valid_len, chunk_start, page_table, bucket, *args, **kwargs)
            self.chunks.append(dict(chunk_start=chunk_start, valid_rows=valid_len, bucket=bucket, retained_rows=piece['rows']))
            self.cursor += valid_len
            return output
        return chunk

    def wrap_slots(self, original):
        # The plugin's _prefill_forward_tp_batched calls prefill_paged_slots, which
        # runs the prompt on a B=1 scratch and writes the result into row
        # empty_slots[0] of the live GDN buffers, leaving the other rows alone. The
        # fast path snapshots, carries and steps at row 0 (gdn_snapshot.ActiveSnapshot),
        # so the second concurrent user's admission read the FIRST user's state and
        # its own sat unread in slot 1 (runs 35492676194, 35493208438). Record the
        # slot so admission can adopt it into slot 0 before anything reads it.
        def slots(token_ids_list, page_table, empty_slots, *args, **kwargs):
            # One CALL per capture, still. A resumed prompt re-enters this on every
            # continuation chunk and will need 'one SLOT per capture' instead - but
            # that relaxation belongs with the change that makes a capture span steps
            # (build plan step 4). Until then this guard is what catches two prompts
            # batched into one step, which the fast path cannot serve, so relaxing it
            # early would remove a live check for a capability that does not exist.
            if not self.active or self.prefill_slot is not None:
                raise ValueError('One batched prefill per capture required')
            self.prefill_slot = prefill_slot(empty_slots)
            return original(token_ids_list, page_table, empty_slots, *args, **kwargs)
        return slots

    @contextmanager
    def capture(self):
        if self.started or self.closed or hasattr(self.model, '_qwen_dflash_prefill_capture') or hasattr(self.model, '_qwen_target_feature_capture'):
            raise ValueError('One non-nested native prefill capture required')
        self.started = self.active = True
        try:
            bindings = [(self.model, '_qwen_dflash_prefill_capture', self),
                        (self.model, '_forward_prefill_chunk_masked_tp', self.wrap(self.model._forward_prefill_chunk_masked_tp))]
            # A model without a batched entry point prefills one sequence through
            # prefill_traced_chunked, whose state lands where the fast path reads it.
            # Every batched entry point the model DOES expose must be wrapped, or its
            # slot goes unrecorded and adopt_prefill_slot reads 'nothing to adopt' -
            # which is a legitimate value for the single-sequence path, so the failure
            # is silent and the next user decodes from this one's recurrent state.
            # That is exactly what M1's prefill_paged_slots_range did: added in
            # lever_n_model_patch, absent from this list, invisible at runtime.
            # Enumerate rather than list, so a method added later fails loudly here
            # instead of silently there.
            for name in sorted(n for n in dir(self.model) if n.startswith('prefill_paged_slots')):
                if not callable(getattr(self.model, name, None)):
                    continue
                if name not in BATCHED_PREFILL_ENTRIES:
                    raise ValueError('Unrecognised batched prefill entry point %r: the capture would '
                                     'not record its GDN slot. Add it to BATCHED_PREFILL_ENTRIES '
                                     'once its empty_slots argument is confirmed third-positional.'
                                     % (name,))
                bindings.append((self.model, name, self.wrap_slots(getattr(self.model, name))))
            with instance_overrides(bindings):
                yield self
            validate_prefill_chunks(self.position, self.chunks)
            self.complete = True
        except BaseException:
            self.active = False
            self.close()
            raise
        finally:
            self.active = False

    def outputs(self):
        if not self.complete or self.active or self.closed:
            raise ValueError('Complete open chunked prefill capture required')
        if self.result is None:
            parts = [child.outputs() for child in self.children]
            output = []
            try:
                for index in range(len(self.taps)):
                    values = [part[index] for part in parts]
                    merged = values[0] if len(values) == 1 else self.operations.concat(values, dim=2,
                        memory_config=self.operations.DRAM_MEMORY_CONFIG)
                    if len(values) > 1:
                        self.merged.append(merged)
                    if merged.shape[2] != self.window['rows']:
                        raise AssertionError('Stitched prefill features must contain exactly the valid tail')
                    output.append(merged)
                self.result = tuple(output)
            except BaseException:
                self.close()
                raise
        return self.result

    def close(self):
        if self.active:
            raise ValueError('Cannot release chunked features during prefill')
        if self.closed:
            return
        self.closed = True
        first_error = None
        for release, value in [(self.operations.deallocate, tensor) for tensor in self.merged] + [
                (lambda captured: captured.close(), child) for child in self.children]:
            try:
                release(value)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        self.merged.clear()
        self.children.clear()
        self.result = None
        if first_error is not None:
            raise first_error
