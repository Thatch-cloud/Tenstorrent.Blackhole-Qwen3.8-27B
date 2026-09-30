"""Single-entry prefix metadata; device checkpoint ownership remains with the caller."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixIdentity:
    target_sha256: str
    drafter_sha256: str
    recipe_sha256: str
    session: str
    allocation_generation: int

    def __post_init__(self):
        for checksum in (self.target_sha256, self.drafter_sha256, self.recipe_sha256):
            if (type(checksum) is not str or len(checksum) != 64
                    or any(character not in '0123456789abcdef' for character in checksum)):
                raise ValueError('Exact model, drafter and runtime/precision fingerprints required')
        if (type(self.session) is not str or not self.session
                or type(self.allocation_generation) is not int or self.allocation_generation < 0):
            raise ValueError('Explicit owning session and allocation generation required')


def tokens_tuple(tokens):
    if (not isinstance(tokens, (tuple, list)) or not 1 <= len(tokens) <= 262144
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError('Bounded exact prompt token IDs required')
    return tuple(tokens)


def owned_pages(pages, inactive_pages, token_count):
    if (not isinstance(pages, (tuple, list)) or not isinstance(inactive_pages, (tuple, list))
            or len(pages) != (token_count + 63) // 64
            or any(type(page) is not int or page < 0 for page in (*pages, *inactive_pages))
            or len(set(pages)) != len(pages) or set(pages).intersection(inactive_pages)):
        raise ValueError('Exclusive valid 64-token physical pages required; omit table padding')
    return tuple(pages)


class PrefixLookup:
    def __init__(self):
        self.invalidate()

    def invalidate(self):
        self.identity = None
        self.tokens = self.pages = ()

    def publish(self, identity, tokens, position, pages, *, inactive_pages=()):
        self.invalidate()
        prompt = tokens_tuple(tokens)
        if (not isinstance(identity, PrefixIdentity) or type(position) is not int
                or not 0 < position < len(prompt) or position % 2048):
            raise ValueError('Completed aligned prompt prefix with an uncached suffix required')
        mapped = owned_pages(pages, inactive_pages, len(prompt))
        self.identity = identity
        self.tokens = prompt[:position]
        self.pages = mapped[:position // 64]

    def match(self, identity, tokens, pages, *, inactive_pages=()):
        prompt = tokens_tuple(tokens)
        mapped = owned_pages(pages, inactive_pages, len(prompt))
        if not isinstance(identity, PrefixIdentity):
            raise ValueError('Explicit cache identity required')
        if self.identity is None:
            return 0
        if identity != self.identity or mapped[:len(self.pages)] != self.pages:
            self.invalidate()
            return 0
        position = len(self.tokens)
        if len(prompt) <= position or prompt[:position] != self.tokens:
            self.invalidate()
            return 0
        return position
