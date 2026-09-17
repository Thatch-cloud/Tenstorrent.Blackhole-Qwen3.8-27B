"""31-query oracle composed from the unchanged single-step native reference."""

from dspark_native_reference import NativeMarkovReference


class T32MarkovReference(NativeMarkovReference):
    def trajectory(self, base_logits, anchor, *, progress=None):
        import torch

        if (not isinstance(base_logits, torch.Tensor) or base_logits.device.type != 'cpu'
                or base_logits.dtype != torch.float32 or tuple(base_logits.shape) != (1, 31, self.vocabulary)
                or not torch.isfinite(base_logits).all() or type(anchor) is not int
                or not 0 <= anchor < self.vocabulary or (progress is not None and not callable(progress))):
            raise ValueError('Finite experimental 31-query logits and valid anchor required')
        native_previous = fp32_previous = anchor
        records = []
        for step in range(31):
            base = base_logits[:, step:step + 1]
            native = super().trajectory(base, native_previous)[0]
            fp32 = native if native_previous == fp32_previous else super().trajectory(base, fp32_previous)[0]
            record = dict(native, fp32_token=fp32['fp32_token'], fp32_previous=fp32_previous,
                fp32_scores=fp32['fp32_scores'])
            records.append(record)
            native_previous, fp32_previous = record['token'], record['fp32_token']
            if progress is not None:
                progress(step)
        return records
