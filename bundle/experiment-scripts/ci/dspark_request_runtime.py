"""DSpark proposal geometry on the existing lossless target-feature publication protocol."""

from dflash_request_runtime import DFlashRequestRuntime


class DSparkRequestRuntime(DFlashRequestRuntime):
    drafter_name = 'dspark'
    proposal_counts = (7, 15)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from publication_diagnostics import PublicationDiagnostics
        self.publication_diagnostics = PublicationDiagnostics()

    def publication_stage(self, name, prefix):
        return self.publication_diagnostics.stage(name, self.position, prefix)
