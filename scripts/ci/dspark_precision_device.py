"""Per-instance proposal precision lifecycle; caller must admit sources before construction."""

from dspark_layer_precision import execute


def device_type(base, native_layer):
    if not isinstance(base, type) or not callable(native_layer):
        raise ValueError('Explicit traced-device type and native proposal backend required')

    class PrecisionDevice(base):
        def __init__(self, *args, **options):
            if options.get('native_attention') is not True:
                raise ValueError('HiFi2 candidate requires native proposal attention')
            super().__init__(*args, **options)
            try:
                if (self.closed or self.prepared is not None
                        or self.proposal_layer is not native_layer or self.max_drafts != 15):
                    raise ValueError('Fresh native fifteen-query proposal device required')
                self.precision_layer_calls = 0

                def proposal_layer(operations, *arguments, **keywords):
                    if self.closed:
                        raise ValueError('Closed precision device cannot execute proposal layers')
                    result = execute(native_layer, operations, *arguments, **keywords)
                    self.precision_layer_calls += 1
                    return result

                self.proposal_layer = proposal_layer
            except BaseException:
                self.close()
                raise

        def propose(self, anchor, count):
            if self.closed or self.prepared is None:
                raise ValueError('Candidate requires its prepared proposal path, never baseline eager fallback')
            return super().propose(anchor, count)

    return PrecisionDevice
