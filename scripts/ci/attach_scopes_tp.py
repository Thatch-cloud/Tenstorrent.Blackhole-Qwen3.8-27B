"""The serving attach's three pair-only scopes, at four cards.

serving_runtime.attach_combined_runtime and dflash_combined_request.combined_runtime enter, for every fast profile,
scopes that were written and qualified for the audited p150a pair. At the pair they stay exactly as they were (their
bytes are hashed by recorded evidence and by test_tp2_pins); at QWEN_FAST_TP=4 tp_addresses.install() rebinds the
three names below to the functions here, so the attach reaches the same call sites and gets four-card behaviour:

  sampler_links            sampling_link_policy.sampler_links refuses any mesh but (1, 2). The attach asks it for the
                           pair's four channels; the ring trains two links per edge, so this scope takes the count
                           mesh_link_policy resolves for the ring (the explicit, audited QWEN_PROJECTION_LINKS) and
                           enters mesh_link_policy.sampler_links, which is the pinned policy's shape for (1, 4).
  scoped_direct_windows    the direct-window hardware batch is the pair's (5,120-wide qkv, (1, 16, 8,240) projections,
                           two chips): its `run` refuses every other 16-row projection, which at four cards is every
                           per-segment GDN call. The four-card profiles switch it off (QWEN_GDN_DIRECT_WINDOW=0) and
                           this scope honours that: it patches nothing, so the native (four-card) batched conv runs,
                           and it refuses a profile that asks for the pair's path at four cards.
  scoped_publication       draft_kv_slide_scope needs QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1 and text-patches the pair's
                           DraftKVHistory. The four-card drafter has its own history class (draft_kv_history_tp) and
                           the profiles switch the slide off, so this scope patches nothing and refuses a profile
                           that asks for it.

Stdlib only at import; mesh_link_policy is imported where used.
"""

from contextlib import contextmanager
import os

# What serving_runtime.attach_combined_runtime passes: the pair's four-channel sampler gather.
PAIR_SAMPLER_LINKS = 4
DIRECT_WINDOW_FLAG = 'QWEN_GDN_DIRECT_WINDOW'
PUBLICATION_FLAG = 'QWEN_DRAFT_KV_SLIDE_EXPERIMENT'


@contextmanager
def sampler_links(sampling, links):
    if links != PAIR_SAMPLER_LINKS:
        raise ValueError('The attach asks for the pair\'s %d sampler links, got %r' % (PAIR_SAMPLER_LINKS, links))
    import mesh_link_policy

    with mesh_link_policy.sampler_links(sampling, mesh_link_policy.projection_links()):
        yield


@contextmanager
def scoped_direct_windows(admission, directory):
    if os.environ.get(DIRECT_WINDOW_FLAG) != '0':
        raise ValueError('%s must be 0 at four cards: the direct-window hardware batch is the pair\'s' % DIRECT_WINDOW_FLAG)
    yield dict(hits=0, fallbacks=0, restored=True, shapes={}, hardware_sources={}, disabled='four-card profile')


@contextmanager
def scoped_publication(directory, evidence):
    if os.environ.get(PUBLICATION_FLAG) != '0':
        raise ValueError('%s must be 0 at four cards: the K/V slide publication is the pair\'s' % PUBLICATION_FLAG)
    yield dict(enabled=False, restored=True, prepare_calls=0, tensor_copies=0, serving_defaults_changed=False,
               admission=None, disabled='four-card profile')
