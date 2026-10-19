"""Prefill lever PF2-H at the model route: the final step hands the GDN state to the decode slot on the DEVICE (QWEN_FAST_PREFILL_HANDOFF=1, default off).

WHAT IS PAID TODAY. The last step of every prompt (levern_route) reads the B=1 prefill scratch to the host (per GDN layer the recurrent state and the K conv taps: 48 x 5 = 240
blocking device-to-host reads) and uploads the same bytes back (240 host-to-device writes, each a tilize on the device) before TPGatedDeltaNet.write_slot puts them into the decode
slot. The device profile of the ship-4 stack (run 38051000905, chip 0, after the last chunk) shows what that costs: the snapshot opens a 48.6 ms gap with the device idle, and the 48-layer
write runs 115.2 ms for 8.0 ms of device kernels (1,919 ops, almost all host dispatch). Every prompt pays it once, and the decode seats wait for it.

WHAT THIS CHANGES. Nothing in the write: the graft's own write_slot runs on the same shapes in the same order. Only its SOURCE changes: instead of tensors uploaded from a host copy
of the scratch, it is handed device-side clones of the scratch (ttnn.clone is a bit copy of the same tensor: same dtype, layout and memory config). The host read-back and the
240 uploads are gone, 240 device copies take their place. The bytes written to the decode slot are the scratch's, as before. All clones are made BEFORE the first write, so a clone that
fails leaves the slot untouched and the host path runs instead (a 'fell back' line, never a failed request); a failure after the first write is raised as it was.

EXACTNESS. By construction: to_torch followed by from_torch to the same dtype is the identity on the values, so the host path writes the scratch's bytes; a clone writes the scratch's
bytes. QWEN_FAST_PREFILL_HANDOFF_AUDIT=1 (gate arms only; the first 8 final steps, then every 24th) proves it on the card: it reads the scratch to the host as the old path does, reads
the whole batched decode state before and after the write, and requires (a) the slot's rows equal the host copy byte for byte on every chip, (b) every other row unchanged.

THE FLAG. QWEN_FAST_PREFILL_HANDOFF: unset = off and nothing here is imported; '0' = off, but the route logs its timing line (the CONTROL arm of the timed pair, so both arms print the
same line); '1' = the device-side hand-off. Strict: anything else raises at the attach. It needs QWEN_FAST_LEVER_N=1 (the route is the only caller). The audit flag needs '1'.
Whatever asks for host bytes (QWEN_FAST_LEVERN_AUDIT, QWEN_PREFIX_AUDIT, QWEN_PREFIX_DIGESTS, this audit) still gets its host snapshot; the hand-off still writes from the device.

MARKERS (the smoke table reads them): '[PINDIAG] prefill handoff engaged ...' once, '... fell back reason=...' on a fallback, '... audit n=.. exact=True' per audited step,
'... audit mismatch' (raises). The timing line '[PF2] handoff mode=<host|device> ms=.. read_ms=.. write_ms=..' on every final step when the flag is set at all.

Stdlib only; ttnn and torch are imported on first use."""

import os
import time

FLAG = 'QWEN_FAST_PREFILL_HANDOFF'
AUDIT_FLAG = 'QWEN_FAST_PREFILL_HANDOFF_AUDIT'
LEVER_N_FLAG = 'QWEN_FAST_LEVER_N'
ENGAGED_MARKER = '[PINDIAG] prefill handoff engaged'
FELL_BACK_MARKER = '[PINDIAG] prefill handoff fell back'
AUDIT_MARKER = '[PINDIAG] prefill handoff audit'
AUDIT_MISMATCH_MARKER = '[PINDIAG] prefill handoff audit mismatch'
TIMING_MARKER = '[PF2] handoff'
AUDIT_FIRST = 8
AUDIT_STRIDE = 24
# Whatever else reads the host snapshot of the scratch on the final step: the handoff keeps that read when one of them is on.
HOST_READERS = ('QWEN_PREFIX_AUDIT', 'QWEN_PREFIX_DIGESTS')


def _flag(name, environ, allowed):
    value = environ.get(name)
    if value is None:
        return None
    if value not in allowed:
        raise ValueError('%s must be %s, got %r' % (name, ' or '.join(repr(item) for item in allowed), value))
    return value


def settings(environ=None):
    """(present, device, audit) for this process; strict. present: the flag is set at all (the route then logs its timing line); device: the hand-off writes from the device."""
    environ = os.environ if environ is None else environ
    value = _flag(FLAG, environ, ('0', '1'))
    audit = _flag(AUDIT_FLAG, environ, ('0', '1'))
    if audit == '1' and value != '1':
        raise ValueError('%s=1 audits the device-side hand-off: it needs %s=1' % (AUDIT_FLAG, FLAG))
    if value is not None and environ.get(LEVER_N_FLAG) != '1':
        raise ValueError('%s is a lever of the Lever N route: it needs %s=1' % (FLAG, LEVER_N_FLAG))
    return value is not None, value == '1', audit == '1'


def requested(environ=None):
    """Whether either flag is present in the environment (anything, so a malformed value still reaches the strict parser and fails at the attach)."""
    environ = os.environ if environ is None else environ
    return FLAG in environ or AUDIT_FLAG in environ


def _ttnn():
    import ttnn

    return ttnn


def _bytes(tensor):
    import torch

    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


class Handoff(object):
    """One process's hand-off: the settings, the latch that a failed clone sets, and the counters the markers and the tests read."""

    def __init__(self, present=True, device=True, audit=False, environ=None):
        self.present, self.device, self.audit = present, device, audit
        self.environ = os.environ if environ is None else environ
        self.latched = None
        self.engaged = 0
        self.fallbacks = 0
        self.calls = 0
        self.audited = 0
        self.mismatches = 0
        self.logged = set()

    @classmethod
    def from_environ(cls, environ=None):
        present, device, audit = settings(environ)
        return cls(present, device, audit, environ)

    # -- what the route asks ------------------------------------------------------------------------------------------------------------
    def active(self):
        """Whether the final step writes the slot from the device (the flag is '1' and no clone has failed)."""
        return self.device and self.latched is None

    def wants_host(self, levern_audit=False):
        """Whether the final step still takes the host snapshot of the scratch: always on the host path, and on the device path when an audit or a digest asks for the bytes."""
        if not self.active():
            return True
        return bool(levern_audit or self.audit or any(self.environ.get(name) == '1' for name in HOST_READERS))

    def audit_due(self):
        if not self.audit or not self.active():
            return False
        index = self.calls
        return index < AUDIT_FIRST or (index - AUDIT_FIRST) % AUDIT_STRIDE == AUDIT_STRIDE - 1

    # -- the timing line of the host path (the control arm) --------------------------------------------------------------------------------
    def log_host(self, log, read_ms, write_ms, layers):
        self.calls += 1
        log('{}', '%s mode=host ms=%.1f read_ms=%.1f write_ms=%.1f layers=%d' % (TIMING_MARKER, read_ms + write_ms, read_ms, write_ms, layers))

    # -- the device path -------------------------------------------------------------------------------------------------------------------
    def clones(self, ttnn, model):
        """[(layer, rec clone, [conv clones])] of the scratch, every clone made before anything is written; raises (after freeing what it made) when one fails."""
        model._ensure_gdn_prefill_scratch()
        made = []
        try:
            for dn, rec, convs, _carry, _zero in model._gdn_prefill_scratch:
                entry = (dn, ttnn.clone(rec, memory_config=ttnn.DRAM_MEMORY_CONFIG), [ttnn.clone(conv, memory_config=ttnn.DRAM_MEMORY_CONFIG) for conv in convs])
                made.append(entry)
        except BaseException:
            for _dn, rec, convs in made:
                for tensor in [rec] + list(convs):
                    try:
                        ttnn.deallocate(tensor)
                    except Exception:
                        pass
            raise
        return made

    def write_slot(self, model, slot, writes, host=None, log=None):
        """Write the scratch's GDN state into decode `slot` from the device. `writes` is the route's _SlotWrites (its count is what tells the lifecycle a slot was written);
        `host` the (rec_snap, conv_snap) the route read, when it did (the audit needs it, and is skipped without it). Returns True when the device path wrote the slot, False when a
        clone failed before anything was written and the caller must run the host path (reading the scratch itself)."""
        log = log or (lambda message, *values: None)
        ttnn = _ttnn()
        began = time.perf_counter()
        before = self.audit_before(ttnn, model) if host is not None and host[0] is not None and self.audit_due() else None
        try:
            made = self.clones(ttnn, model)
        except Exception as error:
            self.latched = '%s: %s' % (type(error).__name__, error)
            self.fallbacks += 1
            log('{}', '%s reason=%s' % (FELL_BACK_MARKER, self.latched))
            return False
        cloned = time.perf_counter()
        for dn, rec, convs in made:
            dn.write_slot(slot, rec, convs)
        writes.count += 1
        done = time.perf_counter()
        self.calls += 1
        self.engaged += 1
        if 'engaged' not in self.logged:
            self.logged.add('engaged')
            log('{}', '%s mode=device layers=%d clones=%d' % (ENGAGED_MARKER, len(made), sum(1 + len(convs) for _dn, _rec, convs in made)))
        log('{}', '%s mode=device ms=%.1f read_ms=0.0 clone_ms=%.1f write_ms=%.1f layers=%d'
            % (TIMING_MARKER, (done - began) * 1000.0, (cloned - began) * 1000.0, (done - cloned) * 1000.0, len(made)))
        if before is not None:
            self.audit_after(ttnn, model, slot, host, before, log)
        return True

    # -- the audit ---------------------------------------------------------------------------------------------------------------------------
    def audit_before(self, ttnn, model):
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        return [(ttnn.to_torch(dn.rec_state, mesh_composer=comp), [ttnn.to_torch(conv, mesh_composer=comp) for conv in dn.conv_states])
                for dn in self.decode_layers(model)]

    @staticmethod
    def decode_layers(model):
        return [layer.attention for layer in model.layers if not layer.is_full_attention]

    def audit_after(self, ttnn, model, slot, host, before, log):
        """Compare the decode buffers after the write with the host copy of the scratch and with themselves before it. `host` is (rec_snap, conv_snap): per layer [chips, ...]
        arrays, the way _read_scratch returns them."""
        rec_snap, conv_snap = host
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        problems = []
        checked = 0
        for li, dn in enumerate(self.decode_layers(model)):
            chips = int(rec_snap[li].shape[0])
            rec_after = ttnn.to_torch(dn.rec_state, mesh_composer=comp)
            rec_before, conv_before = before[li]
            per_chip = int(rec_after.shape[0]) // chips
            for chip in range(chips):
                rows = rec_after[chip * per_chip:(chip + 1) * per_chip]
                was = rec_before[chip * per_chip:(chip + 1) * per_chip]
                if _bytes(rows[slot:slot + 1]) != _bytes(rec_snap[li][chip:chip + 1]):
                    problems.append('layer %d chip %d recurrent slot %d differs from the scratch' % (li, chip, slot))
                if _bytes(_without(rows, slot)) != _bytes(_without(was, slot)):
                    problems.append('layer %d chip %d recurrent: a row other than slot %d changed' % (li, chip, slot))
                checked += 1
            for tap, conv in enumerate(dn.conv_states):
                after = ttnn.to_torch(conv, mesh_composer=comp)
                was = conv_before[tap]
                for chip in range(chips):
                    if _bytes(after[chip:chip + 1, slot:slot + 1]) != _bytes(conv_snap[li][tap][chip:chip + 1]):
                        problems.append('layer %d chip %d conv tap %d slot %d differs from the scratch' % (li, chip, tap, slot))
                    if _bytes(_without(after[chip], slot, dim=0)) != _bytes(_without(was[chip], slot, dim=0)):
                        problems.append('layer %d chip %d conv tap %d: a row other than slot %d changed' % (li, chip, tap, slot))
                    checked += 1
        self.audited += 1
        if problems:
            self.mismatches += 1
            log('{}', '%s n=%d exact=False slot=%d problems=%d first=%s' % (AUDIT_MISMATCH_MARKER, self.audited, slot, len(problems), problems[0]))
            raise AssertionError('%s: %s' % (AUDIT_MISMATCH_MARKER, '; '.join(problems[:3])))
        log('{}', '%s n=%d exact=True slot=%d layers=%d tensors_checked=%d' % (AUDIT_MARKER, self.audited, slot, len(self.decode_layers(model)), checked))


def _without(tensor, index, dim=0):
    import torch

    kept = [i for i in range(int(tensor.shape[dim])) if i != index]
    return tensor.index_select(dim, torch.as_tensor(kept, dtype=torch.long))
