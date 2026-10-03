"""Single-request opt-in prepared-proposal hook; no serving default changes."""

from contextlib import contextmanager
from pathlib import Path

from dspark_markov_device import execute as native
from dspark_markov_score_layout import execute as candidate
from dspark_score_layout_hardware_gate import qualify, validate_hardware
from gdn_multitoken_conv import addresses


class ScoreLayoutArm:
    def __init__(self, device, *, hardware_audit=None):
        if device.closed or device.max_drafts != 15 or list(device.mesh.shape) != [1, 2]:
            raise ValueError('Open fifteen-query two-chip drafter required')
        self.device = device
        self.calls = 0
        self.installed = self.restored = self.failed = False
        self.qualification = None
        self.hardware_audit = hardware_audit

    @contextmanager
    def install(self, prepared_module=None):
        if self.installed or self.restored or self.failed:
            raise ValueError('Fresh single-use score-layout scope required')
        if prepared_module is None:
            import dspark_prepared_proposal as prepared_module
        if prepared_module.markov is not native:
            raise ValueError('Unmodified native prepared Markov binding required')
        self.qualification = qualify(Path(__file__).parent)
        self.hardware_digest = validate_hardware(self.hardware_audit, Path(__file__).parent)
        if self.hardware_audit.get('weight_bindings') != [addresses(self.device.operations, weight)
                for weight in (self.device.predecessor, self.device.successor)]:
            raise ValueError('Hardware audit must belong to this loaded learned-weight allocation')

        def feedback(operations, anchor, logits, predecessor, successor, owned, *, on_step_enqueued=None):
            device = self.device
            if (device.closed or operations is not device.operations or predecessor is not device.predecessor
                    or successor is not device.successor or tuple(logits.shape) != (1, 1, 15, 248320)):
                self.failed = True
                raise ValueError('Only this request owner and complete fifteen-query vocabulary may use the hook')
            try:
                result = candidate(operations, device.mesh, anchor, logits, predecessor, successor, owned,
                    on_step_enqueued=on_step_enqueued)
            except BaseException:
                self.failed = True
                raise
            self.calls += 1
            return result

        prepared_module.markov = feedback
        self.installed = True
        try:
            yield self
        except BaseException:
            self.failed = True
            raise
        finally:
            unchanged = prepared_module.markov is feedback
            if unchanged:
                prepared_module.markov = native
                self.restored = True
            self.installed = False
            if not unchanged:
                self.failed = True
                raise RuntimeError('Score-layout hook changed externally; refusing to overwrite another owner')

    def summary(self):
        if self.failed or self.installed or not self.restored or self.calls < 1:
            raise ValueError('Successful used and restored score-layout scope required')
        return dict(calls=self.calls, restored=True, qualification=self.qualification,
            hardware_audit=self.hardware_audit, hardware_audit_sha256=self.hardware_digest,
            scope='Prepared Markov score layout only; request correctness and speed require separate measurement')
