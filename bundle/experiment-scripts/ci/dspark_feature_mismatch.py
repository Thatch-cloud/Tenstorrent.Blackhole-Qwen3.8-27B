"""Exact feature mismatch evidence without relaxing the acceptance check."""

from dspark_projection import tensor_digest


class FeatureMismatch(AssertionError):
    def __init__(self, actual, expected, *, tap, chip, position):
        import torch

        self.evidence = dict(kind='committed_target_feature', tap=tap, chip=chip, position=position,
            actual_shape=list(actual.shape), expected_shape=list(expected.shape),
            actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
            actual_sha256=tensor_digest(actual), expected_sha256=tensor_digest(expected),
            actual_finite=bool(torch.isfinite(actual).all()), expected_finite=bool(torch.isfinite(expected).all()))
        if actual.shape == expected.shape:
            mismatch = actual != expected
            self.evidence['mismatched_elements'] = int(mismatch.sum())
            coordinates = mismatch.nonzero()
            if len(coordinates):
                coordinate = tuple(int(value) for value in coordinates[0])
                self.evidence['first_coordinate'] = list(coordinate)
                self.evidence['first_actual'] = str(float(actual[coordinate]))
                self.evidence['first_expected'] = str(float(expected[coordinate]))
            self.evidence['max_abs'] = str(float((actual.float() - expected.float()).abs().max()))
        super().__init__(f'DSpark committed feature mismatch: {self.evidence}')
