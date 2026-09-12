"""Lossless target-publication bridge for a parallel learned DFlash2 proposer."""

from contextlib import nullcontext

TARGET_TAPS = (5, 19, 33, 47, 61)


class DFlashRequestRuntime:
    tap_ids = TARGET_TAPS
    drafter_name = 'dflash2'
    proposal_counts = (7, 31)

    def __init__(self, drafter, *, position, validate_features=None):
        if (type(position) is not int or position < 1 or drafter.position != position
                or not all(callable(getattr(drafter, name, None)) for name in ('propose', 'prepare_publication', 'commit_publication', 'discard_publication'))
                or (validate_features is not None and not callable(validate_features))):
            raise ValueError('Prepared feature drafter at the exact prefilled frontier required')
        self.drafter, self.position = drafter, position
        self.max_drafts = getattr(drafter, 'max_drafts', 7)
        if type(self.max_drafts) is not int or self.max_drafts not in self.proposal_counts:
            raise ValueError('Explicit supported feature-drafter proposal geometry required')
        self.validate_features = validate_features
        self.session = self.engine = None
        self.phase = 'unbound'
        self.proposed = ()
        self.committed_feature_rows = 0

    def bind(self, session, engine):
        if (self.phase != 'unbound' or session.phase != 'idle' or session.position != self.position
                or tuple(engine.retain_feature_taps) != self.tap_ids or engine.session is not session):
            raise ValueError('Current prefilled request and complete feature-retaining verifier required')
        self.session, self.engine = session, engine
        self.phase = 'idle'

    def __call__(self, request_id, history, count):
        if (self.phase != 'idle' or self.session.request_id != request_id
                or self.session.phase != 'drafting' or self.session.position != self.position
                or self.drafter.position != self.position or not history or history[-1] != self.session.seed
                or type(count) is not int or count < 1):
            raise ValueError('Feature proposal requires the current committed request frontier')
        self.phase = 'drafting'
        try:
            candidates = tuple(self.drafter.propose(self.session.seed, min(count, self.max_drafts)))
            if (len(candidates) != min(count, self.max_drafts) or any(type(token) is not int or not 0 <= token < self.session.vocab_size
                    for token in candidates)):
                raise ValueError('Complete global feature-drafter proposal IDs required')
            self.proposed = candidates
            self.phase = 'proposed'
            return candidates
        except BaseException:
            self.phase = 'failed'
            raise

    def publish(self, prefix):
        ticket = self.session.pending
        if (self.phase not in ('idle', 'proposed') or ticket is None or self.session.phase != 'committing'
                or self.engine.phase != 'verified' or self.engine.pending is not ticket
                or ticket.position != self.position or type(prefix) is not int or not 0 <= prefix <= len(ticket.tokens)):
            raise ValueError('Feature publication requires the current verified target transaction')
        self.phase = 'committing'
        publication = None
        try:
            if prefix:
                with self.publication_stage('features', prefix):
                    features = self.engine.verified_features_for_publication(ticket)
                    if self.validate_features is not None:
                        self.validate_features(features, prefix, self.position)
                with self.publication_stage('prepare_history', prefix):
                    publication = self.drafter.prepare_publication(features, prefix, position=self.position)
            with self.publication_stage('publish_target', prefix):
                self.engine.publish(prefix)
            if publication is not None:
                with self.publication_stage('commit_history', prefix):
                    self.drafter.commit_publication(publication)
            if self.drafter.position != self.position + prefix:
                raise ValueError('Drafter and target feature frontiers diverged')
            self.position += prefix
            self.committed_feature_rows += prefix
            self.proposed = ()
            self.phase = 'idle'
        except BaseException:
            self.phase = self.engine.phase = 'failed'
            if publication is not None:
                self.drafter.discard_publication(publication)
            raise

    def publication_stage(self, name, prefix):
        return nullcontext()
