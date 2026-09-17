"""Explicit output budgets for matched full-request experiments."""


def request_limit(override=None, *, short_default=False):
    if override is not None:
        if type(override) is not int or not 2 <= override <= 257:
            raise ValueError('Request output limit must be an integer from 2 through 257')
        return override
    return 256 if short_default else 257
