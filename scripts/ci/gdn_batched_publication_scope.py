"""Explicit simulator-gated T16 publication hook; no automatic hardware or serving use."""

from contextlib import contextmanager
from pathlib import Path

from gdn_commit_dma import prepare as native
from gdn_commit_batched_dma import prepare as candidate, validate_shapes
from gdn_batched_publication_gate import qualify


class PublicationArm:
    def __init__(self, mesh, native_layers):
        if tuple(mesh.shape) != (1, 2) or len(native_layers) != 48 or any(len(layer) != 5 for layer in native_layers):
            raise ValueError('Both chips and all 48 native GDN state owners required')
        self.mesh = mesh
        self.native_layers = [tuple(layer) for layer in native_layers]
        self.prepared_prefixes = []
        self.native_fallbacks = 0
        self.installed = self.restored = self.failed = False

    @contextmanager
    def install(self, verifier_module=None):
        if self.installed or self.restored or self.failed:
            raise ValueError('Fresh single-use publication scope required')
        if verifier_module is None:
            import verifier_engine as verifier_module
        if verifier_module.prepare is not native:
            raise ValueError('Unmodified native publication binding required')
        self.qualification = qualify(Path(__file__).parent)

        def prepare(mesh, layers, prefix):
            try:
                if mesh is not self.mesh or len(layers) != 48:
                    raise ValueError('Only the declared mesh and complete model may use this hook')
                rows = validate_shapes([[tuple(value.shape) for value in layer] for layer in layers], prefix)
                if any(any(actual is not expected for actual, expected in zip(layer[10:15], owned, strict=True))
                        for layer, owned in zip(layers, self.native_layers, strict=True)):
                    raise ValueError('Native state ownership changed')
                if rows != 16:
                    self.native_fallbacks += 1
                    return native(mesh, layers, prefix)
                operation = candidate(mesh, layers, prefix)
                self.prepared_prefixes.append(prefix)
                return operation
            except BaseException:
                self.failed = True
                raise

        verifier_module.prepare = prepare
        self.installed = True
        try:
            yield self
        except BaseException:
            self.failed = True
            raise
        finally:
            unchanged = verifier_module.prepare is prepare
            if unchanged:
                verifier_module.prepare = native
                self.restored = True
            self.installed = False
            if not unchanged:
                self.failed = True
                raise RuntimeError('Publication hook changed externally; refusing to overwrite another owner')

    def summary(self):
        if self.failed or self.installed or not self.restored or set(self.prepared_prefixes) != set(range(17)):
            raise ValueError('Successful restored scope with every T16 publication prefix required')
        return dict(qualification=self.qualification, prepared_prefixes=list(self.prepared_prefixes),
            native_fallbacks=self.native_fallbacks, restored=True, hardware_qualified=False,
            serving_qualified=False, scope='Preparation route only; learned hardware continuation and timed requests remain required')
