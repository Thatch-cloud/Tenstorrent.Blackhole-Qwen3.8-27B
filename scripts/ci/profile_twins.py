"""Every generated profile twin the profile-enumerating tests exempt (each generator's own test holds its twins): engine reuse's (make_parked_profiles),
round-host's (make_round_host_profiles), octo-T8's (make_octo_profiles), the tp4/octo-2 levers' (make_octo2_profiles), the W2 kill drill's (make_w2_kill_profiles), the
region-read audit's (make_kvread_profiles) and the host KV tier's (make_prefix_tier_profiles). One list, so a lever branch that adds twins adds them here and not to a dozen tests."""

import make_kvread_profiles
import make_octo2_profiles
import make_octo_profiles
import make_parked_profiles
import make_prefix_tier_profiles
import make_round_host_profiles
import make_w2_kill_profiles


def twin_names():
    return (tuple(make_parked_profiles.twin_names()) + tuple(make_round_host_profiles.twin_names())
            + tuple(make_octo_profiles.twin_names()) + tuple(make_octo2_profiles.twin_names())
            + tuple(make_w2_kill_profiles.twin_names()) + tuple(make_kvread_profiles.twin_names())
            + tuple(make_prefix_tier_profiles.twin_names()))
