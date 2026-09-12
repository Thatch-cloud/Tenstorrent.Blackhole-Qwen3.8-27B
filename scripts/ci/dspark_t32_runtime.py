"""Experimental T32 publication adapter; no simulator or hardware admission implied."""

from dspark_request_runtime import DSparkRequestRuntime


class T32DSparkRequestRuntime(DSparkRequestRuntime):
    proposal_counts = (31,)
