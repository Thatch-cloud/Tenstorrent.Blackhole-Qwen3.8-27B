"""Unqualified two-bank proposal traces; no request route or serving default selects this class."""

from dspark_history import leaves
from dspark_prepared_proposal import PreparedDSparkProposal
from gdn_multitoken_conv import addresses


def binding(operations, bank):
    if len(bank) != 5 or any(len(pair) != 2 for pair in bank):
        raise ValueError('Five complete history K/V pairs required')
    result = tuple(addresses(operations, value) for value in leaves(bank))
    if any(len(pair) != 2 for pair in result):
        raise ValueError('Two-chip history storage required')
    return result


class BoundProposal(PreparedDSparkProposal):
    def __init__(self, device, anchor, bank, *, audit, defer_capture=False):
        self.bank = bank
        self.bank_binding = binding(device.operations, bank)
        if self.bank_binding not in tuple(binding(device.operations, value)
                for value in (device.history.layers, device.history.spare_layers)):
            raise ValueError('Trace must borrow an existing history-owned bank')
        super().__init__(device, anchor, audit=audit, defer_capture=defer_capture)

    def allocate_history(self):
        return self.bank

    def update_history(self):
        if binding(self.operations, self.device.history.layers) != self.bank_binding:
            raise ValueError('Only the committed bank bound to this trace may replay')


class BankedDSparkProposal:
    def __init__(self, device, anchor, *, audit=False):
        if device.closed or device.history.pending is not None or type(audit) is not bool:
            raise ValueError('Idle live history and explicit audit policy required')
        self.device, self.closed, self.checks = device, False, []
        self.banks = (device.history.layers, device.history.spare_layers)
        self.bindings = tuple(binding(device.operations, bank) for bank in self.banks)
        for chip in range(2):
            pointers = [pair[chip] for bank in self.bindings for pair in bank]
            if len(set(pointers)) != 20:
                raise ValueError('Both complete history banks must have independent storage')
        self.proposals = []
        self.replay_counts = [0, 0]
        try:
            for bank in self.banks:
                self.proposals.append(BoundProposal(device, anchor, bank, audit=audit, defer_capture=True))
            for proposal in self.proposals:
                proposal.capture()
        except BaseException:
            self.close()
            raise

    def propose(self, anchor, count):
        if self.closed or self.device.closed or self.device.history.pending is not None:
            raise ValueError('Live committed bank required')
        if tuple(binding(self.device.operations, bank) for bank in self.banks) != self.bindings:
            raise ValueError('Captured bank addresses changed')
        active = binding(self.device.operations, self.device.history.layers)
        spare = binding(self.device.operations, self.device.history.spare_layers)
        if active not in self.bindings or spare != self.bindings[1 - self.bindings.index(active)]:
            raise ValueError('Active and spare banks must be the captured complementary pair')
        index = self.bindings.index(active)
        proposal = self.proposals[index]
        before = len(proposal.checks)
        result = proposal.propose(anchor, count)
        self.checks.extend(proposal.checks[before:])
        self.replay_counts[index] += 1
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        first_error = None
        for proposal in reversed(self.proposals):
            try:
                proposal.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
