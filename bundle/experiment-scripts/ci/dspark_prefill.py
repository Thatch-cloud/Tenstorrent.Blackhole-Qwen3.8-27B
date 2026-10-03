"""Owned native prefill chunks for full-history DSpark, without a sliding window."""

from contextlib import contextmanager
from dataclasses import dataclass

from dspark_intake import TAPS
from gdn_multitoken_conv import addresses
from model_batch import instance_overrides
from target_features import LayerOutputCapture


@dataclass(frozen=True)
class FeatureChunk:
    start: int
    rows: int
    features: tuple


def validate_chunks(chunks, *, start, rows):
    if (type(start) is not int or start < 0 or type(rows) is not int or rows < 1
            or not isinstance(chunks, (tuple, list)) or not chunks):
        raise ValueError('Nonempty full-history chunks and explicit absolute bounds required')
    cursor = start
    for chunk in chunks:
        if (not isinstance(chunk, FeatureChunk) or type(chunk.start) is not int or chunk.start != cursor
                or type(chunk.rows) is not int or chunk.rows < 1 or len(chunk.features) != len(TAPS)):
            raise ValueError('All five taps in complete ordered, contiguous native chunks required')
        shape = tuple(chunk.features[0].shape)
        if (len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != 2560 or shape[2] % 32
                or not chunk.rows <= shape[2] <= 2048
                or any(tuple(value.shape) != shape for value in chunk.features)):
            raise ValueError('Matching padded native TP2 prefill outputs required')
        cursor += chunk.rows
    if cursor != start + rows:
        raise ValueError('Full-history chunks must cover every valid position exactly once')


class FullHistoryCapture:
    def __init__(self, operations, model, position):
        if (type(position) is not int or not 1 <= position <= 8192
                or not callable(getattr(model, '_forward_prefill_chunk_masked_tp', None))):
            raise ValueError('Pinned native prefill boundary and full history up to8192 required')
        self.operations, self.model, self.position = operations, model, position
        self.children, self.chunks = [], []
        self.cursor = 0
        self.started = self.active = self.complete = self.closed = False

    def snapshot(self, value, bucket):
        from dspark_projection import require_tensor

        require_tensor(self.operations, value, (1, 1, bucket, 2560), self.operations.bfloat16)
        return self.operations.clone(value, memory_config=self.operations.DRAM_MEMORY_CONFIG)

    def wrap(self, original):
        def chunk(token_buf, valid_len, chunk_start, page_table, bucket, *args, **kwargs):
            if (not self.active or type(chunk_start) is not int or chunk_start != self.cursor
                    or type(valid_len) is not int or valid_len < 1
                    or type(bucket) is not int or bucket % 32 or not valid_len <= bucket <= 2048
                    or chunk_start + valid_len > self.position):
                raise ValueError('One complete ordered pass over bounded native prefill chunks required')
            captured = LayerOutputCapture(self.model, TAPS,
                snapshot=lambda value: self.snapshot(value, bucket), release=self.operations.deallocate,
                storage_ids=lambda value: tuple(enumerate(addresses(self.operations, value))))
            self.children.append(captured)
            with captured.capture():
                output = original(token_buf, valid_len, chunk_start, page_table, bucket, *args, **kwargs)
            self.chunks.append(FeatureChunk(chunk_start, valid_len, captured.outputs()))
            self.cursor += valid_len
            return output
        return chunk

    @contextmanager
    def capture(self):
        markers = ('_qwen_dspark_prefill_capture', '_qwen_dflash_prefill_capture', '_qwen_target_feature_capture')
        if self.started or self.closed or any(hasattr(self.model, name) for name in markers):
            raise ValueError('One non-nested full-history prefill capture required')
        self.started = self.active = True
        try:
            with instance_overrides([(self.model, '_qwen_dspark_prefill_capture', self),
                    (self.model, '_forward_prefill_chunk_masked_tp', self.wrap(self.model._forward_prefill_chunk_masked_tp))]):
                yield self
            validate_chunks(self.chunks, start=0, rows=self.position)
            self.complete = True
        except BaseException:
            self.active = False
            self.close()
            raise
        finally:
            self.active = False

    def outputs(self):
        if not self.complete or self.active or self.closed:
            raise ValueError('Complete open full-history capture required')
        return tuple(self.chunks)

    def close(self):
        if self.active:
            raise ValueError('Cannot release features during prefill')
        if self.closed:
            return
        self.closed = True
        children, self.children = self.children, []
        self.chunks.clear()
        first_error = None
        for child in children:
            try:
                child.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
