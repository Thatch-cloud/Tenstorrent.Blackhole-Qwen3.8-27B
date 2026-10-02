"""The A0 screen's target side: raw taps from a BF16 forward over the logged text, chunked, with prefix sharing inside a
conversation. Python / torch; the real model is a Hugging Face Qwen3.5 text model on the GPU host (HFTarget, canary-verified);
everything else here runs on CPU against a tiny hybrid model (a0_fakes.TinyTarget).

THE TARGET INTERFACE (what GroupRunner needs; HFTarget and the fake implement it):
    taps                    a TapRecorder over the model's decoder layers
    new_state()             a fresh cache / recurrent state
    forward(state, ids)     [n] int ids -> the final hidden rows [n, H]; advances the state by n rows; the taps of those n rows
                            are waiting in `taps`
    snapshot(state)         the recurrent and convolution state and the attention length (NOT the keys and values: a 122k-token
                            KV snapshot would double the cache)
    restore(state, snap)    back to the snapshot (the attention cache is cropped to the recorded length)
    argmax(hidden)          greedy token per row (the LM head; chunked)

RAW TAPS. The drafters read `hidden_states[i + 1]` of decoder layer i for i in 5, 19, 33, 47, 61 (post-layer residual), concatenated
in tap order: 51,200 B per row at 5120 wide in BF16. TapRecorder takes them from forward hooks, so no full hidden-state tuple is built.

PREFIX SHARING (GroupRunner). Turns of one conversation whose prompts extend each other (a0_bundle's prefix groups) are traced once:
prefill the prompt up to turn k's length, snapshot, forward turn k's answer (all but its last token), restore, continue the prefill
to turn k+1's length. Rows of the prompt are identical for every later turn (causal), so one buffer serves the group.

V1. The target's own argmax over the answer rows against the logged answer: row P - 1 predicts answer[0], row P + i predicts
answer[i + 1]. Counts only; the first divergence offset per turn (the screen reports it).
"""
import torch

CHUNK = 4096


class TapRecorder(object):
    """Forward hooks on decoder layers `tap_ids` of `layers` (an indexable of modules). Each hooked layer's output (a tensor, or
    a tuple whose first item is the hidden state) is kept for the chunk in flight."""

    def __init__(self, layers, tap_ids, dtype=None):
        self.layers, self.tap_ids, self.dtype = layers, tuple(tap_ids), dtype
        self.latest, self.handles = {}, []

    def attach(self):
        for index in self.tap_ids:
            def hook(module, inputs, output, index=index):
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.latest[index] = value.detach()
            self.handles.append(self.layers[index].register_forward_hook(hook))
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def collect(self):
        """[n, taps * H] for the last forward, in tap order; clears the buffer."""
        if set(self.latest) != set(self.tap_ids):
            raise RuntimeError('a tap layer did not run')
        rows = torch.cat([self.latest[index].reshape(-1, self.latest[index].shape[-1]) for index in self.tap_ids], dim=-1)
        self.latest = {}
        return rows.to(self.dtype) if self.dtype is not None else rows


class FeatureBuffer(object):
    """A [rows, width] tensor filled chunk by chunk."""

    def __init__(self, rows, width, dtype, device='cpu'):
        self.data = torch.empty(rows, width, dtype=dtype, device=device)
        self.filled = 0

    def write(self, start, rows):
        if start != self.filled:
            raise ValueError('rows are written in order')
        self.data[start:start + rows.shape[0]] = rows
        self.filled += rows.shape[0]

    def rows(self, a, b):
        if b > self.filled:
            raise ValueError('rows not written yet')
        return self.data[a:b]


class FeatureView(object):
    """Rows of one turn: the shared prompt buffer below `prompt_len`, the turn's own answer rows above it."""

    def __init__(self, prompt_buffer, prompt_len, answer_rows):
        self.prompt, self.prompt_len, self.answer = prompt_buffer, prompt_len, answer_rows

    def __len__(self):
        return self.prompt_len + self.answer.shape[0]

    def rows(self, a, b):
        if a < 0 or b > len(self) or a > b:
            raise ValueError('rows out of range')
        if b <= self.prompt_len:
            return self.prompt.rows(a, b)
        if a >= self.prompt_len:
            return self.answer[a - self.prompt_len:b - self.prompt_len]
        return torch.cat([self.prompt.rows(a, self.prompt_len), self.answer[:b - self.prompt_len]], dim=0)


class V1(object):
    """The target-argmax check of one turn: rows compared, rows that agree, the first disagreeing answer offset (or None)."""

    def __init__(self):
        self.rows = self.agree = 0
        self.first_divergence = None

    def add(self, predicted, expected, base):
        for at, (p, e) in enumerate(zip(predicted, expected)):
            self.rows += 1
            if p == e:
                self.agree += 1
            elif self.first_divergence is None:
                self.first_divergence = base + at

    def as_dict(self):
        return dict(rows=self.rows, agree=self.agree, first_divergence=self.first_divergence)


class GroupRunner(object):
    """Runs the turns of one prefix group through the target and hands each turn's features to `consume`."""

    def __init__(self, target, chunk=CHUNK, feature_dtype=torch.bfloat16, feature_device='cpu'):
        self.target, self.chunk = target, chunk
        self.dtype, self.device = feature_dtype, feature_device

    def _chunks(self, ids, a, b):
        for start in range(a, b, self.chunk):
            yield start, min(b, start + self.chunk)

    def run_group(self, records, consume, width, wanted=None):
        """`records`: the group's turns in `order` (each: prompt_ids, output_ids; prompts extend each other). `consume(record,
        view, v1)` is called once per turn with that turn's FeatureView; `wanted(record)` false skips the answer pass and
        the call for that turn (its prompt still extends the shared trace). Returns nothing; the buffers are released after."""
        target = self.target
        longest = max(len(record['prompt_ids']) for record in records)
        buffer = FeatureBuffer(longest, width, self.dtype, self.device)
        state = target.new_state()
        done, last_hidden, previous = 0, None, []
        for record in records:
            prompt, answer = record['prompt_ids'], record['output_ids']
            if len(prompt) < len(previous) or list(prompt[:len(previous)]) != list(previous):
                raise ValueError('the prompts of a prefix group must extend each other')
            previous = prompt
            for a, b in self._chunks(prompt, done, len(prompt)):
                ids = torch.as_tensor(prompt[a:b], dtype=torch.int64)
                last_hidden = target.forward(state, ids)[-1:]
                buffer.write(a, target.taps.collect().to(self.dtype).to(self.device))
            done = max(done, len(prompt))
            if wanted is not None and not wanted(record):
                continue
            v1 = V1()
            v1.add(target.argmax(last_hidden).tolist(), [answer[0]], 0)
            snapshot = target.snapshot(state)
            taps = []
            body = answer[:-1]                                  # rows P .. P + L - 2
            for a, b in self._chunks(body, 0, len(body)):
                ids = torch.as_tensor(body[a:b], dtype=torch.int64)
                hidden = target.forward(state, ids)
                taps.append(target.taps.collect().to(self.dtype).to(self.device))
                v1.add(target.argmax(hidden).tolist(), answer[a + 1:b + 1], a + 1)
            rows = torch.cat(taps, dim=0) if taps else torch.empty(0, width, dtype=self.dtype, device=self.device)
            consume(record, FeatureView(buffer, len(prompt), rows), v1)
            del taps, rows
            target.restore(state, snapshot)
        del buffer, state


def branch_selfcheck(target, prompt, answer_a, answer_b, width, chunk=CHUNK, atol=1e-4):
    """The canary's own test of snapshot / restore: prefill, branch into answer A, restore, branch into B, and compare B's taps and
    argmax with a fresh prefill of prompt + B. True when they agree (per-row argmax equal, taps within `atol` relative)."""
    def trace(ids):
        state, rows, preds = target.new_state(), [], []
        for a in range(0, len(ids), chunk):
            hidden = target.forward(state, torch.as_tensor(ids[a:a + chunk], dtype=torch.int64))
            rows.append(target.taps.collect().float())
            preds.extend(target.argmax(hidden).tolist())
        return torch.cat(rows), preds

    state = target.new_state()
    for a in range(0, len(prompt), chunk):
        target.forward(state, torch.as_tensor(prompt[a:a + chunk], dtype=torch.int64))
        target.taps.collect()
    snapshot = target.snapshot(state)
    for branch in (answer_a, answer_b):
        taps, preds = [], []
        for a in range(0, len(branch), chunk):
            hidden = target.forward(state, torch.as_tensor(branch[a:a + chunk], dtype=torch.int64))
            taps.append(target.taps.collect().float())
            preds.extend(target.argmax(hidden).tolist())
        target.restore(state, snapshot)
    branch_taps, branch_preds = torch.cat(taps), preds
    fresh_taps, fresh_preds = trace(list(prompt) + list(answer_b))
    fresh_taps, fresh_preds = fresh_taps[len(prompt):], fresh_preds[len(prompt):]
    scale = float(fresh_taps.abs().max()) or 1.0
    return branch_preds == fresh_preds and float((branch_taps - fresh_taps).abs().max()) <= atol * scale


class HFTarget(object):
    """A Hugging Face Qwen3.5 text model (Qwen3_5ForCausalLM) as a TargetModel. GPU HOST ONLY: nothing here is exercised on CPU, and
    it is trusted only after the canary's V0 / V1 and `branch_selfcheck` pass. The recurrent-state attribute names of the cache
    layers are listed in RECURRENT_ATTRS; a transformers version that names them differently makes snapshot() raise, never guess."""
    RECURRENT_ATTRS = ('conv_states', 'recurrent_states')

    def __init__(self, model, tap_ids, device, head_chunk=512):
        self.model, self.base, self.device, self.head_chunk = model, model.model, device, head_chunk
        self.taps = TapRecorder(self.base.layers, tap_ids).attach()

    def new_state(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.model.config)

    @torch.no_grad()
    def forward(self, state, ids):
        out = self.base(input_ids=ids[None].to(self.device), past_key_values=state, use_cache=True)
        return out.last_hidden_state[0]

    def snapshot(self, state):
        saved = []
        for index, layer in enumerate(state.layers):
            tensors = dict((name, getattr(layer, name).clone()) for name in self.RECURRENT_ATTRS
                           if getattr(layer, name, None) is not None)
            saved.append((index, tensors))
        if not any(tensors for _, tensors in saved):
            raise RuntimeError('no recurrent state found in the cache layers: the cache layout is not the one expected')
        return dict(length=state.get_seq_length(), layers=saved)

    def restore(self, state, snapshot):
        state.crop(snapshot['length'])
        for index, tensors in snapshot['layers']:
            for name, value in tensors.items():
                getattr(state.layers[index], name).copy_(value)

    @torch.no_grad()
    def argmax(self, hidden):
        head = self.model.lm_head.weight
        return torch.cat([(hidden[a:a + self.head_chunk] @ head.T).argmax(-1) for a in range(0, hidden.shape[0], self.head_chunk)]).cpu()
