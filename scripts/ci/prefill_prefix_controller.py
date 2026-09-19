"""Opt-in single-session orchestration; caller must hold an exclusive KV-page lease."""

from contextlib import contextmanager

from dspark_prefill import FullHistoryCapture, validate_chunks
from prefill_prefix_boundary import checkpoint_boundary
from prefill_prefix_features import prefix_features
from prefill_prefix_lookup import PrefixLookup, tokens_tuple
from prefill_prefix_resume import native_resume_scope


class PrefixController:
    def __init__(self, operations, model, checkpoint, validate_residency):
        if not callable(validate_residency):
            raise ValueError('Explicit live KV residency validator required')
        self.operations, self.model, self.checkpoint = operations, model, checkpoint
        self.validate_residency = validate_residency
        self.lookup, self.owner, self.busy = PrefixLookup(), None, False

    def invalidate(self):
        if self.busy:
            raise ValueError('Cannot release a prefix during its active request')
        self.lookup.invalidate()
        owner, self.owner = self.owner, None
        if owner is not None:
            owner.close()

    def retain_prefix(self, position):
        owner = self.owner
        retained = [chunk for chunk in owner.chunks if chunk.start + chunk.rows <= position]
        validate_chunks(retained, start=0, rows=position)
        if len(owner.children) != len(owner.chunks):
            raise ValueError('Cold capture must own every retained feature chunk')
        removed = owner.children[len(retained):]
        owner.children = owner.children[:len(retained)]
        owner.chunks = retained
        owner.position = owner.cursor = position
        first_error = None
        for child in removed:
            try:
                child.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    @contextmanager
    def request(self, identity, tokens, pages, prefill, *, prefix_position, inactive_pages):
        if self.busy or not callable(prefill):
            raise ValueError('Exactly one active prefix-cache request required')
        import torch

        if (not isinstance(tokens, torch.Tensor) or tokens.device.type != 'cpu'
                or tokens.ndim != 2 or tokens.shape[0] != 1
                or tokens.dtype not in (torch.int32, torch.int64)
                or not isinstance(pages, torch.Tensor) or pages.device.type != 'cpu'
                or pages.ndim != 2 or pages.shape[0] != 1
                or pages.dtype not in (torch.int32, torch.int64)
                or type(prefix_position) is not int or not 0 < prefix_position < tokens.shape[1]
                or prefix_position % 2048):
            raise ValueError('Aligned text-only host request required')
        token_ids = tokens[0].tolist()
        valid_pages = pages[0, :(len(token_ids) + 63) // 64].tolist()
        try:
            self.validate_residency(identity, valid_pages, inactive_pages)
            position = self.lookup.match(identity, token_ids, valid_pages, inactive_pages=inactive_pages)
            if position and position != prefix_position:
                position = 0
            if not position:
                self.invalidate()
            self.busy = True
            if position:
                with prefix_features(self.operations, self.model, len(token_ids), self.owner, position) as capture:
                    with native_resume_scope(self.operations, self.model, tokens, pages,
                            prefix_position=position, restore=self.checkpoint.restore) as route:
                        with capture.capture():
                            result = prefill(tokens)
                    yield result, capture, dict(cache_hit=True, prefix_tokens=position,
                        suffix_tokens=len(token_ids) - position, native_route=route)
            else:
                self.owner = FullHistoryCapture(self.operations, self.model, len(token_ids))
                with checkpoint_boundary(self.model, self.checkpoint, prefix_position) as boundary:
                    with self.owner.capture():
                        result = prefill(tokens)
                self.validate_residency(identity, valid_pages, inactive_pages)
                self.lookup.publish(identity, token_ids, prefix_position, valid_pages,
                    inactive_pages=inactive_pages)
                yield result, self.owner, dict(cache_hit=False, prefix_tokens=0,
                    suffix_tokens=len(token_ids), checkpoint_boundary=boundary)
                self.retain_prefix(prefix_position)
        except BaseException:
            self.busy = False
            self.invalidate()
            raise
        finally:
            self.busy = False


def full_request_factory(controller, identity, pages, *, prefix_position, inactive_pages):
    import torch

    page_table = pages.clone()
    inactive = tuple(inactive_pages)

    @contextmanager
    def factory(tokens, prefill, ordinal, *, prime_tokens=None):
        if type(ordinal) is not int or ordinal not in (0, 1):
            raise ValueError('One cold control and one cached candidate required')
        requested = tokens_tuple(tokens)
        priming = None
        if prime_tokens is not None:
            prime = tokens_tuple(prime_tokens)
            if (ordinal != 1 or len(prime) != len(requested)
                    or prime[:prefix_position] != requested[:prefix_position]
                    or prime[prefix_position:] == requested[prefix_position:]):
                raise ValueError('Changed-suffix priming requires the exact same prefix and request length')
            controller.invalidate()
            with controller.request(identity, torch.tensor([prime], dtype=torch.int32), page_table,
                    lambda values: prefill(values[0].tolist()), prefix_position=prefix_position,
                    inactive_pages=inactive) as (unused, unused_capture, prime_record):
                if prime_record['cache_hit'] is not False:
                    raise ValueError('Changed-suffix priming must be cold')
                priming = dict(changed_suffix=True, prefix_tokens=prefix_position,
                    prompt_tokens=len(prime), checkpoint_boundary=prime_record['checkpoint_boundary'])
        if ordinal == 0:
            controller.invalidate()
        host_tokens = torch.tensor([requested], dtype=torch.int32)
        with controller.request(identity, host_tokens, page_table,
                lambda values: prefill(values[0].tolist()), prefix_position=prefix_position,
                inactive_pages=inactive) as result:
            if priming is not None:
                result[2]['changed_suffix_priming'] = priming
            yield result

    return factory
