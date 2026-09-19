"""Unqualified drafter-only HiFi2 projection candidate; never a target policy."""

from dspark_layer import linear as reference_linear


class ProjectionOperations:
    def __init__(self, operations):
        self.operations = operations
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.operations, name)

    def matmul(self, value, weight, **options):
        original = options.get('compute_kernel_config')
        fields = dict(math_fidelity=self.operations.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        if original is None or any(getattr(original, name, None) != expected for name, expected in fields.items()):
            raise ValueError('Unchanged HiFi4 FP32 drafter projection reference required')
        if options.get('dtype') != self.operations.float32:
            raise ValueError('FP32 projection output must be retained')
        options['compute_kernel_config'] = self.operations.WormholeComputeKernelConfig(
            **dict(fields, math_fidelity=self.operations.MathFidelity.HiFi2))
        self.calls += 1
        return self.operations.matmul(value, weight, **options)


def linear(operations, value, weight, retain, *, rounded=True):
    selected = ProjectionOperations(operations)
    result = reference_linear(selected, value, weight, retain, rounded=rounded)
    if selected.calls != 1:
        raise ValueError('Exactly one explicitly selected drafter projection required')
    return result
