"""Opt-in request runtime for the validated T8, norm-batched, four-link candidate."""

import os

from sampling_link_policy import audit, sampler_links
from verifier_engine import VerifierEngine


PROFILE = 'p150-pair-t8-norm-four-link-v1'
EVIDENCE_RUN = 34185881285


class CandidateRuntime:
    def __init__(self, model, session, pages, helpers, *, sampler, norm_batch=True,
                 attention_replay=False, attention_mask_once=False, replay_group_rows=4, max_verify_rows=8):
        if (norm_batch is not True or attention_replay is not False or attention_mask_once is not False
                or type(replay_group_rows) is not int or replay_group_rows != 4
                or type(max_verify_rows) is not int or max_verify_rows != 8):
            raise ValueError('Candidate profile requires fixed native attention, batched norm and T8 cap')
        self.sources = audit('/opt/tt-metal', os.environ)
        self.engine = None
        self.closed = False
        self.scope = sampler_links(sampler.tt_sampling, 4)
        self.scope.__enter__()
        try:
            self.engine = VerifierEngine(model, session, pages, helpers, sampler=sampler,
                norm_batch=True, max_verify_rows=8)
        except BaseException:
            self.closed = True
            self.scope.__exit__(None, None, None)
            raise

    @property
    def buckets(self):
        return self.engine.buckets

    @property
    def setup_ms(self):
        return self.engine.setup_ms

    @property
    def phase(self):
        return 'closed' if self.closed else self.engine.phase

    def verify(self, ticket):
        if self.closed:
            raise RuntimeError('Candidate runtime is closed')
        return self.engine.verify(ticket)

    def publish(self, prefix):
        if self.closed:
            raise RuntimeError('Candidate runtime is closed')
        return self.engine.publish(prefix)

    def close(self):
        if self.closed:
            return
        try:
            self.engine.close()
        finally:
            self.closed = True
            self.scope.__exit__(None, None, None)
