"""Host-only bounded capture plan; not yet connected to the request engine."""

from dataclasses import dataclass

from attention_mask_replay import validate_ticket
from verifier_engine import capture_widths


@dataclass(frozen=True)
class Capture:
    rows: int
    position: int
    capacity: int | None

    @property
    def key(self):
        return self.rows, self.capacity


@dataclass(frozen=True)
class ReplayPlan:
    start: int
    stop: int
    captures: tuple[Capture, ...]

    def available(self, position, remaining):
        if any(type(value) is not int for value in (position, remaining)):
            raise ValueError('Integer request position and remaining budget required')
        if not self.start <= position < self.stop or not 1 <= remaining <= self.stop - position:
            raise ValueError('Ticket is outside the prepared request budget')
        family = (position // 256 + 1) * 256
        return tuple(capture for capture in self.captures if capture.rows <= remaining and
            (capture.capacity is None or (capture.capacity == family and position + capture.rows <= family)))

    def max_rows(self, position, remaining):
        return max(capture.rows for capture in self.available(position, remaining))

    def select(self, position, rows, remaining):
        if type(rows) is not int:
            raise ValueError('Integer ticket width required')
        matches = [capture for capture in self.available(position, remaining) if capture.rows == rows]
        if len(matches) != 1:
            raise ValueError('Ticket has no unique prepared native or mask-family bucket')
        return matches[0]


def capture_plan(position, page_capacity, verifier_rows, remaining, *, max_verify_rows=32, short_context=False):
    widths = capture_widths(position, page_capacity, verifier_rows, remaining, max_verify_rows)
    stop = position + remaining
    if type(short_context) is not bool or (short_context and
            (max_verify_rows != 8 or position < 128 or stop > 768)):
        raise ValueError('Short-context routing requires a T8 cap and complete budget in positions 128..767')
    captures = [Capture(rows, position, None) for rows in widths if rows < 8]
    for capacity in range((position // 256 + 1) * 256, ((stop - 1) // 256 + 1) * 256 + 1, 256):
        first = max(position, capacity - 256)
        last = min(stop, capacity)
        for rows in widths:
            if rows < 8 or first + rows > last:
                continue
            validate_ticket(first, rows, capacity, short_context=short_context)
            captures.append(Capture(rows, first, capacity))
    if len({capture.key for capture in captures}) != len(captures):
        raise AssertionError('Capture keys must be unique')
    return ReplayPlan(position, stop, tuple(captures))
