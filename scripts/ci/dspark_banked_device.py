"""Experimental bank-bound drafting device; never selected by serving defaults."""

from dspark_banked_proposal import BankedDSparkProposal
from dspark_prepared_proposal import TracedDSparkDevice


class BankedDSparkDevice(TracedDSparkDevice):
    def prepare_trace(self, anchor, *, audit=False):
        if self.closed or self.prepared is not None:
            raise ValueError('One prepared proposal trace set per live drafter required')
        self.prepared = BankedDSparkProposal(self, anchor, audit=audit)
