"""Tiny stand-ins for the A0 screen's heavy parts, shared by the tests: a hybrid target (recurrent layers and attention layers, as
Qwen3.5 mixes them), a drafter that follows the logged answer, and a feature source. No weights, no text."""
import torch
from torch import nn

import a0_target as tgt


class LinearBlock(nn.Module):
    """A gated linear recurrence with a 3-tap causal convolution: state = (h, the last two inputs). Sequential per token, so any
    chunking of the same rows gives the same numbers."""
    kind = 'linear'

    def __init__(self, hidden, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.taps = nn.Parameter(0.5 * torch.randn(3, hidden, generator=generator))
        self.gate = nn.Parameter(torch.randn(hidden, generator=generator))
        self.out = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.out.weight.copy_(0.3 * torch.randn(hidden, hidden, generator=generator))

    def forward(self, x, layer):
        h, tail = layer['h'], layer['tail']
        outputs = []
        for t in range(x.shape[1]):
            row = x[0, t]
            conv = self.taps[0] * row + self.taps[1] * tail[0] + self.taps[2] * tail[1]
            h = torch.sigmoid(self.gate) * h + conv
            tail = torch.stack([row, tail[0]])
            outputs.append(x[0, t] + self.out(torch.tanh(h)))
        layer['h'], layer['tail'] = h, tail
        return torch.stack(outputs)[None]


class AttentionBlock(nn.Module):
    kind = 'attn'

    def __init__(self, hidden, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(0.4 * torch.randn(3 * hidden, hidden, generator=generator))
            self.out.weight.copy_(0.3 * torch.randn(hidden, hidden, generator=generator))

    def forward(self, x, layer):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        keys = k if layer['k'] is None else torch.cat([layer['k'], k], dim=1)
        values = v if layer['v'] is None else torch.cat([layer['v'], v], dim=1)
        layer['k'], layer['v'] = keys, values
        total, new = keys.shape[1], x.shape[1]
        scores = q @ keys.transpose(1, 2) / (x.shape[-1] ** 0.5)
        allowed = torch.arange(total)[None, :] <= (total - new + torch.arange(new))[:, None]
        scores = scores.masked_fill(~allowed[None], float('-inf'))
        return x + self.out(torch.softmax(scores, dim=-1) @ values)


class TinyHybrid(nn.Module):
    def __init__(self, vocab=50, hidden=16, kinds=('linear', 'attn', 'linear', 'attn', 'linear', 'attn'), seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, hidden, generator=generator))
            self.lm_head.weight.copy_(torch.randn(vocab, hidden, generator=generator))
        self.layers = nn.ModuleList([(LinearBlock if kind == 'linear' else AttentionBlock)(hidden, seed * 100 + at)
                                     for at, kind in enumerate(kinds)])

    def new_cache(self):
        return [dict(h=torch.zeros(self.embed.embedding_dim), tail=torch.zeros(2, self.embed.embedding_dim)) if layer.kind == 'linear'
                else dict(k=None, v=None) for layer in self.layers]

    def forward(self, ids, cache, return_all=False):
        x = self.embed(ids)[None]
        every = []
        for layer, state in zip(self.layers, cache):
            x = layer(x, state)
            every.append(x)
        return (x, every) if return_all else x


class TinyTarget(object):
    """tgt's TargetModel over TinyHybrid."""

    def __init__(self, model, tap_ids):
        self.model = model.eval()
        self.taps = tgt.TapRecorder(model.layers, tap_ids).attach()

    def new_state(self):
        return dict(cache=self.model.new_cache())

    @torch.no_grad()
    def forward(self, state, ids):
        return self.model(ids, state['cache'])[0]

    def snapshot(self, state):
        saved = []
        for layer in state['cache']:
            if 'h' in layer:
                saved.append(dict(h=layer['h'].clone(), tail=layer['tail'].clone()))
            else:
                saved.append(dict(length=0 if layer['k'] is None else layer['k'].shape[1]))
        return saved

    def restore(self, state, snapshot):
        for layer, saved in zip(state['cache'], snapshot):
            if 'h' in layer:
                layer['h'], layer['tail'] = saved['h'].clone(), saved['tail'].clone()
            elif layer['k'] is not None:
                layer['k'], layer['v'] = layer['k'][:, :saved['length']], layer['v'][:, :saved['length']]

    @torch.no_grad()
    def argmax(self, hidden):
        return (hidden @ self.model.lm_head.weight.T).argmax(-1)


class FollowsAnswer(object):
    """A drafter that proposes the logged answer's next tokens for the first `good` positions, then a token that is wrong."""
    name = 'follows'

    def __init__(self, good=15, wrong=-5):
        self.good, self.wrong, self.answer, self.prompt = good, wrong, None, 0

    def begin(self, record, features):
        self.answer, self.prompt = record['output_ids'], len(record['prompt_ids'])

    def propose(self, sequence, start, count):
        index = start - self.prompt
        out = []
        for at in range(count):
            where = index + 1 + at
            out.append(self.answer[where] if at < self.good and where < len(self.answer) else self.wrong)
        return out

    def end(self):
        self.answer = None
