"""Explicit output budgets for matched full-request experiments."""


def request_limit(override=None, *, t32=False, short_default=False):
    if override is not None:
        if type(override) is not int or not 2 <= override <= 257:
            raise ValueError('Request output limit must be an integer from 2 through 257')
        if t32 and override != 65:
            raise ValueError('T32 integration currently qualifies only the 65-token output budget')
        return override
    return 65 if t32 else 256 if short_default else 257
