"""The seeded sampled arm of the test clients (the serving smoke, the tau lab): stdlib only, host side, nothing here is served.

A profile that turns sampled verification on (QWEN_FAST_SAMPLED=1, docs/tp4-sampled-verify.md) honours a request's temperature, top_p, top_k and seed. This module is the one place the clients
decide what a "sampled" request looks like, so two runs on two profiles are comparable:

  greedy         (the default, every client as it always was) no sampling field is added: the request is whatever the client built, which the greedy contract serves as greedy.
  model-default  the model's own defaults (generation_config.json of the base snapshot: temperature 1.0, top_p 0.95, top_k 20) and a FIXED SEED per request.

The seed is a pure function of the arm's base seed (C2_SMOKE_SEED / the lab's --sampling-seed, default 20261022) and the request's PROMPT: the same prompt always carries the same seed, in any run,
on any profile, whatever the budget, the test it appears in or the order it is sent. So a re-sent prompt re-draws the same text (the server keys its noise by (seed, position) only), a longer
budget answer begins with the shorter budget's, and the same request sent to a sampled profile and to a plain-sampling reference gets the same seed. Requests that are not completions (tokenize,
models) are left alone, and a request that names its own temperature, top_p, top_k or seed keeps it. The smoke's own `agreement` (log-probabilities) and `bench` (fixed shapes) tests build their
requests elsewhere (tp_agreement, tp_decode_bench) and stay greedy: logprobs are refused under sampling.
"""

import hashlib
import json

MODES = ('greedy', 'model-default')
DEFAULT_MODE = 'greedy'
MODEL_DEFAULTS = dict(temperature=1.0, top_p=0.95, top_k=20)      # sampled_accept.MODEL_DEFAULTS (held equal by a test; this module is stdlib only)
DEFAULT_BASE_SEED = 20261022
COMPLETION_PATHS = ('/v1/chat/completions', '/v1/completions')
SEED_BITS = 63                                                   # vLLM takes an int64 seed


class SamplingError(ValueError):
    pass


def mode_of(value):
    """'greedy' or 'model-default'; empty or None is the default (greedy). Anything else is a configuration error, never a silent greedy."""
    if value is None or value == '':
        return DEFAULT_MODE
    if value not in MODES:
        raise SamplingError('sampling must be one of %s, got %r' % (', '.join(MODES), value))
    return value


def base_seed_of(value):
    if value is None or value == '':
        return DEFAULT_BASE_SEED
    try:
        seed = int(str(value), 10)
    except ValueError:
        raise SamplingError('the sampling base seed is a decimal integer, got %r' % (value,)) from None
    if not 0 <= seed < (1 << SEED_BITS):
        raise SamplingError('the sampling base seed is 0..2^%d - 1, got %r' % (SEED_BITS, value))
    return seed


def prompt_key(body):
    """The material a request's seed is drawn from: its prompt (the chat messages with the tools, or the prompt text / token ids), canonical JSON. Not its budget, stream flag or name."""
    material = dict(messages=body.get('messages'), tools=body.get('tools'), prompt=body.get('prompt'))
    return json.dumps(material, sort_keys=True, separators=(',', ':'), default=list)


def seed_for(base, body):
    digest = hashlib.sha256(('%d|%s' % (base, prompt_key(body))).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') & ((1 << SEED_BITS) - 1)


class Arm(object):
    """One client's sampling arm: its mode and base seed; apply() adds the terms to a completion request body."""

    def __init__(self, mode=DEFAULT_MODE, base_seed=DEFAULT_BASE_SEED):
        self.mode, self.base_seed = mode_of(mode), base_seed_of(base_seed)
        self.sent = 0

    @classmethod
    def from_environ(cls, environ, mode_name='C2_SMOKE_SAMPLING', seed_name='C2_SMOKE_SEED'):
        return cls(environ.get(mode_name), environ.get(seed_name))

    @property
    def sampled(self):
        return self.mode != 'greedy'

    def terms(self, body):
        """The sampling fields for `body` ({} for the greedy arm)."""
        if not self.sampled:
            return {}
        return dict(MODEL_DEFAULTS, seed=seed_for(self.base_seed, body))

    def apply(self, body, path=None):
        """`body` with the arm's terms (a copy; the client's own fields win: a request that names a temperature, top_p, top_k or seed keeps it). The greedy arm and a path that is not
        a completion return `body` itself."""
        if not self.sampled or (path is not None and path not in COMPLETION_PATHS):
            return body
        out = dict(body)
        for name, value in self.terms(body).items():
            out.setdefault(name, value)
        self.sent += 1
        return out

    def report(self, counted=True):
        """What SMOKE_JSON (and the lab's summary) says about the arm. `counted` adds the number of requests apply() gave the terms (the smoke's; the lab reports before it sends)."""
        return dict(mode=self.mode, base_seed=self.base_seed, **({'requests': self.sent} if counted else {}),
                    **({} if not self.sampled else dict(terms=dict(MODEL_DEFAULTS), seed_rule='sha256(base_seed|prompt)[:8], 63 bits')))
