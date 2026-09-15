"""Transactional dirty-tile plan for an experimental incremental history writer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppendPlan:
    position: int
    prefix: int
    first_row: int
    end_row: int


class BankAppendPlanner:
    def __init__(self, position, capacity):
        if (type(position) is not int or type(capacity) is not int
                or not 0 <= position <= capacity or capacity < 32 or capacity % 32):
            raise ValueError('Valid frontier and tile-aligned capacity required')
        self.position, self.capacity = position, capacity
        self.repair_start = self.dirty_end = position
        self.pending = None

    def prepare(self, prefix):
        if (self.pending is not None or type(prefix) is not int or not 1 <= prefix <= 32
                or self.position + prefix > self.capacity):
            raise ValueError('One bounded append transaction required')
        first = self.repair_start // 32 * 32
        end = (max(self.dirty_end, self.position + prefix) + 31) // 32 * 32
        self.pending = AppendPlan(self.position, prefix, first, end)
        return self.pending

    def resolve(self, plan, *, commit):
        if plan is not self.pending or self.pending is None or type(commit) is not bool:
            raise ValueError('Current transaction identity and explicit resolution required')
        if commit:
            self.repair_start = self.position
            self.position += plan.prefix
            self.dirty_end = self.position
        else:
            self.repair_start = self.position
            self.dirty_end = self.position + plan.prefix
        self.pending = None
