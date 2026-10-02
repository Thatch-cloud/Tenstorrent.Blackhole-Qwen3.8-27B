"""WP-F1: what one drafter fine-tune example is on disk (Python 3.7 + numpy; no torch import at module level).

AN EXAMPLE is one agent turn of the target: its context ids, the target's own GREEDY answer ids (S2 is lossless greedy, so the answer
is what plain greedy would write), and the five target tap rows for the 2,048 rows before the answer plus the answer rows themselves
(anchors sit in the answer and each block reads its 2,048-row window). Only counts and opaque ids leave this module's schema: the ids
are lab-store data, never public.

FEATURE FILES hold the tap rows, row-major, as raw bf16 bit patterns (uint16):
    kind 'raw'     5 taps x 5120 wide = 25,600 columns, 51,200 B per row   (needed if fc trains)
    kind 'postfc'  hidden_norm(fc(taps)) = 5,120 columns, 10,240 B per row (valid only while fc and hidden_norm stay frozen)
    kind 'raw_bf8' 27,200 B per row (bfloat8_b taps; training fidelity unproven: sized, not written, here)
Layout: magic `FTF1`, a uint32 header length, a JSON header padded to 64 bytes (kind, rows, width, payload_offset, payload_sha256),
then the payload. `FeatureFile` memory-maps the payload; a size or sha256 mismatch is refused.

TP4 ASSEMBLY. Each chip holds 1,280 columns of EVERY tap, laid out tap-major locally: [tap0 chip slice, tap1 chip slice, ...]. The
global row is tap-major over the whole width: tap t = the chips' tap-t slices in chip order. Joining the chips' rows end to end puts
chip 0's five taps first and is the wrong feature order (a tested mutation): `assemble_taps` interleaves per tap.
"""
import hashlib
import json
import os
import struct

import numpy as np

MAGIC = b'FTF1'
HIDDEN = 5120
TAPS = 5
CHIPS = 4
PER_CHIP = HIDDEN // CHIPS               # 1,280 columns of each tap per chip
WINDOW = 2048
KINDS = {'raw': (TAPS * HIDDEN, 2), 'postfc': (HIDDEN, 2)}
BYTES_PER_ROW = {'raw': TAPS * HIDDEN * 2, 'postfc': HIDDEN * 2, 'raw_bf8': 27200}
SOURCES = ('swe', 'chained', 'code', 'own')
ALIGN = 64
EXAMPLE_FIELDS = ('example_id', 'source', 'context_ids', 'answer_ids', 'feature_file', 'kind', 'finish', 'censored', 'weight')


class DatasetError(ValueError):
    """A file or an example that does not meet the format; the message names a field or a count, never a value of the data."""


# -- tap assembly ---------------------------------------------------------------------------------------------------------------

def assemble_taps(chip_rows, taps=TAPS, per_chip=PER_CHIP):
    """[chip arrays [rows, taps * per_chip]] -> [rows, taps * chips * per_chip], tap-major over the global width."""
    chips = len(chip_rows)
    if not chips or any(rows.ndim != 2 or rows.shape != chip_rows[0].shape or rows.shape[1] != taps * per_chip for rows in chip_rows):
        raise DatasetError('every chip needs the same rows x (taps * per_chip) shape')
    rows = chip_rows[0].shape[0]
    out = np.empty((rows, taps * chips * per_chip), dtype=chip_rows[0].dtype)
    for tap in range(taps):
        for chip, mine in enumerate(chip_rows):
            start = (tap * chips + chip) * per_chip
            out[:, start:start + per_chip] = mine[:, tap * per_chip:(tap + 1) * per_chip]
    return out


def split_taps(rows, chips=CHIPS, taps=TAPS):
    """The inverse of assemble_taps: [rows, taps * width] -> a list of per-chip [rows, taps * per_chip] arrays."""
    width = rows.shape[1] // taps
    if rows.ndim != 2 or rows.shape[1] != taps * width or width % chips:
        raise DatasetError('the rows do not split into %d chips of whole tap slices' % chips)
    per_chip = width // chips
    return [np.concatenate([rows[:, tap * width + chip * per_chip: tap * width + (chip + 1) * per_chip] for tap in range(taps)], axis=1)
            for chip in range(chips)]


# -- bf16 bits ------------------------------------------------------------------------------------------------------------------

def bf16_bits(values):
    """float32 -> bf16 bit patterns (uint16), round to nearest even; NaN stays NaN."""
    raw = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = raw + 0x7FFF + ((raw >> 16) & 1)
    bits = (rounded >> 16).astype(np.uint16)
    nan = np.isnan(values)
    if nan.any():
        bits[nan] = 0x7FC0
    return bits


def bf16_values(bits):
    """bf16 bit patterns (uint16) -> float32."""
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# -- the feature file -----------------------------------------------------------------------------------------------------------

def write_features(path, kind, bits):
    """Write [rows, width] uint16 bf16 bit patterns as a feature file of `kind` ('raw' or 'postfc')."""
    if kind not in KINDS:
        raise DatasetError('kind must be one of %s' % ', '.join(sorted(KINDS)))
    width, size = KINDS[kind]
    if bits.dtype != np.uint16 or bits.ndim != 2 or bits.shape[1] != width:
        raise DatasetError('%s rows are uint16 [rows, %d]' % (kind, width))
    payload = np.ascontiguousarray(bits).tobytes()
    header = dict(kind=kind, rows=int(bits.shape[0]), width=width, dtype='bf16', payload_sha256=hashlib.sha256(payload).hexdigest())
    # the offset depends on the header length, which holds the offset: iterate to a fixed point (two rounds)
    offset = 0
    for _ in range(3):
        header['payload_offset'] = offset
        text = json.dumps(header, sort_keys=True).encode('utf-8')
        offset = -(-(len(MAGIC) + 4 + len(text)) // ALIGN) * ALIGN
    header['payload_offset'] = offset
    text = json.dumps(header, sort_keys=True).encode('utf-8')
    if len(MAGIC) + 4 + len(text) > offset:
        raise DatasetError('the header does not fit its padding')
    with open(path, 'wb') as handle:
        handle.write(MAGIC + struct.pack('<I', len(text)) + text)
        handle.write(b'\0' * (offset - handle.tell()))
        handle.write(payload)
    return header


class FeatureFile(object):
    """A memory-mapped feature file. `rows(a, b)` is a uint16 view; nothing is copied."""

    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as handle:
            head = handle.read(len(MAGIC) + 4)
            if head[:len(MAGIC)] != MAGIC or len(head) != len(MAGIC) + 4:
                raise DatasetError('not a feature file')
            (length,) = struct.unpack('<I', head[len(MAGIC):])
            if length > 4096:
                raise DatasetError('the header is larger than a header can be')
            try:
                self.header = json.loads(handle.read(length).decode('utf-8'))
            except ValueError:
                raise DatasetError('the header is not JSON')
        for name in ('kind', 'rows', 'width', 'payload_offset', 'payload_sha256'):
            if name not in self.header:
                raise DatasetError('the header has no %s' % name)
        if self.header['kind'] not in KINDS or self.header['width'] != KINDS[self.header['kind']][0]:
            raise DatasetError('the header names a kind or width that does not exist')
        self.kind, self.count, self.width = self.header['kind'], self.header['rows'], self.header['width']
        if os.path.getsize(path) != self.header['payload_offset'] + self.count * self.width * 2:
            raise DatasetError('the file size does not match its header')
        self.data = np.memmap(path, dtype=np.uint16, mode='r', offset=self.header['payload_offset'], shape=(self.count, self.width))

    def rows(self, a, b):
        if not 0 <= a <= b <= self.count:
            raise DatasetError('rows out of range')
        return self.data[a:b]

    def verify(self, block=1 << 24):
        """Recompute the payload sha256 (streaming) and compare it with the header's."""
        digest = hashlib.sha256()
        flat = self.data.reshape(-1).view(np.uint8) if self.count else np.empty(0, np.uint8)
        for at in range(0, flat.shape[0], block):
            digest.update(flat[at:at + block].tobytes())
        if digest.hexdigest() != self.header['payload_sha256']:
            raise DatasetError('the payload does not match its sha256')
        return True

    def close(self):
        self.data = None


# -- the example schema -----------------------------------------------------------------------------------------------------------

def expected_rows(prompt_len, answer_len, window=WINDOW):
    """Rows a feature file holds for an example: the window of prompt rows before the answer, then the answer rows (every answer
    token but the last: the last token is only ever predicted, never an input)."""
    return min(window, prompt_len) + max(0, answer_len - 1)


def validate_example(example, window=WINDOW):
    """Refuse an example that does not meet the schema. -> the example."""
    if not isinstance(example, dict) or any(name not in example for name in EXAMPLE_FIELDS):
        raise DatasetError('an example carries exactly the schema fields: %s' % ', '.join(EXAMPLE_FIELDS))
    if example['source'] not in SOURCES:
        raise DatasetError('source must be one of %s' % ', '.join(SOURCES))
    for name in ('context_ids', 'answer_ids'):
        ids = example[name]
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise DatasetError('%s must be a non-empty list of non-negative integers' % name)
    if example['kind'] not in KINDS or not isinstance(example['censored'], bool):
        raise DatasetError('kind must be raw or postfc and censored a boolean')
    if not isinstance(example['weight'], (int, float)) or example['weight'] <= 0:
        raise DatasetError('weight must be positive')
    if example['finish'] not in ('stop', 'length', 'tool_calls'):
        raise DatasetError('finish must be stop, length or tool_calls')
    if example['censored'] != (example['finish'] == 'length'):
        raise DatasetError('an answer cut by its budget (finish length) is censored, and only that one')
    return example


def check_features(example, features, window=WINDOW):
    """The example's feature file must hold exactly the rows the schema says, of its kind."""
    want = expected_rows(len(example['context_ids']), len(example['answer_ids']), window)
    if features.kind != example['kind'] or features.count != want:
        raise DatasetError('the feature file does not match the example (kind or rows)')
    return True


# -- sizes -----------------------------------------------------------------------------------------------------------------------

def storage_per_turn(kind, prompt_tokens, answer_tokens, window=WINDOW):
    """Bytes one example's features take."""
    if kind not in BYTES_PER_ROW:
        raise DatasetError('unknown kind')
    return expected_rows(prompt_tokens, answer_tokens, window) * BYTES_PER_ROW[kind]


def tier_bytes(kind, turns, prompt_tokens=35000, answer_tokens=260, window=WINDOW):
    return turns * storage_per_turn(kind, prompt_tokens, answer_tokens, window)
