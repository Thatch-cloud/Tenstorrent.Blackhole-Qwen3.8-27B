"""Independent Blackhole grouped-product oracle for an explicitly approximate DSpark proposal policy."""

from projection_rounding import grouped_projection_reference


POLICY = 'Blackhole grouped BF16 products with FP32 scores; proposal-only'


def audit_scores(scores, token, reference):
    import torch

    expected = reference['scores']
    if (scores.dtype != torch.float32 or scores.shape != expected.shape or not torch.isfinite(scores).all()
            or not torch.equal(scores.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))
            or type(token) is not int or token != reference['token']):
        raise AssertionError('Native Markov scores and greedy token must exactly match the grouped-product oracle')

    def diagnostic(other):
        close = torch.isclose(scores, other, rtol=1e-4, atol=1e-4)
        return dict(max_abs=float((scores - other).abs().max()), mismatched=int((~close).sum()),
            full_vocabulary_close=bool(close.all()), token=int(other.argmax(-1)[0]))

    return dict(token=token, token_exact=True, full_vocabulary_exact=True, max_abs=0.,
        previous=reference['previous'], fp32_previous=reference['fp32_previous'],
        same_input_fp32=diagnostic(reference['same_input_fp32_scores']),
        fp32_trajectory=diagnostic(reference['fp32_scores']))


class NativeMarkovReference:
    def __init__(self, predecessor, successor, *, chunk_columns=2048):
        import torch

        if (any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.bfloat16
                for value in (predecessor, successor)) or predecessor.ndim != 2 or successor.shape != predecessor.shape
                or predecessor.shape[1] != 256 or not 32 <= predecessor.shape[0] <= 248320
                or predecessor.shape[0] % 32 or type(chunk_columns) is not int
                or not 32 <= chunk_columns <= 4096 or chunk_columns % 32
                or any(not torch.isfinite(value).all() for value in (predecessor, successor))):
            raise ValueError('Finite CPU BF16 rank256 matrices and bounded tile-aligned reference chunks required')
        self.predecessor = predecessor.clone()
        self.successor = successor.clone()
        self.successor_fp32 = successor.float()
        self.vocabulary, self.chunk_columns = predecessor.shape[0], chunk_columns
        self.native_cache, self.fp32_cache = {}, {}

    def bias(self, token, *, native):
        import torch

        if type(token) is not int or not 0 <= token < self.vocabulary or type(native) is not bool:
            raise ValueError('Explicit in-vocabulary predecessor and arithmetic selection required')
        cache = self.native_cache if native else self.fp32_cache
        if token not in cache:
            feature = self.predecessor[token:token + 1]
            if native:
                chunks = [grouped_projection_reference(feature, self.successor[start:start + self.chunk_columns].T,
                    destination_rounding=True, fidelity_span=32).float()
                    for start in range(0, self.vocabulary, self.chunk_columns)]
                result = torch.cat(chunks, dim=-1)
            else:
                result = feature.float() @ self.successor_fp32.T
            if not torch.isfinite(result).all():
                raise ValueError('Nonfinite Markov reference score')
            cache[token] = result
        return cache[token]

    def trajectory(self, base_logits, anchor, *, progress=None):
        import torch

        if (not isinstance(base_logits, torch.Tensor) or base_logits.device.type != 'cpu'
                or base_logits.dtype != torch.float32 or base_logits.ndim != 3
                or base_logits.shape[0] != 1 or base_logits.shape[2] != self.vocabulary
                or not 1 <= base_logits.shape[1] <= 15 or not torch.isfinite(base_logits).all()
                or type(anchor) is not int or not 0 <= anchor < self.vocabulary
                or (progress is not None and not callable(progress))):
            raise ValueError('Finite single-sequence FP32 base logits and valid anchor required')
        native_previous = fp32_previous = anchor
        result = []
        for step in range(base_logits.shape[1]):
            base = base_logits[:, step]
            native_scores = base + self.bias(native_previous, native=True)
            same_input_fp32 = base + self.bias(native_previous, native=False)
            fp32_scores = base + self.bias(fp32_previous, native=False)
            if any(not torch.isfinite(value).all() for value in (native_scores, same_input_fp32, fp32_scores)):
                raise ValueError('Markov reference addition overflowed')
            native_token, fp32_token = int(native_scores.argmax(-1)[0]), int(fp32_scores.argmax(-1)[0])
            result.append(dict(token=native_token, previous=native_previous, scores=native_scores,
                same_input_fp32_scores=same_input_fp32, fp32_token=fp32_token, fp32_previous=fp32_previous,
                fp32_scores=fp32_scores))
            native_previous, fp32_previous = native_token, fp32_token
            if progress is not None:
                progress(step)
        return result
