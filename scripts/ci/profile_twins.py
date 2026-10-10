"""Every generated profile twin the profile-enumerating tests exempt (each generator's own test holds its twins): engine reuse's (make_parked_profiles),
round-host's (make_round_host_profiles), octo-T8's (make_octo_profiles) and the W2 kill drill's (make_w2_kill_profiles). One list, so a lever branch that adds twins adds them here and not to a dozen tests."""

import make_octo_profiles
import make_parked_profiles
import make_round_host_profiles
import make_w2_kill_profiles


def twin_names():
    return (tuple(make_parked_profiles.twin_names()) + tuple(make_round_host_profiles.twin_names())
            + tuple(make_octo_profiles.twin_names()) + tuple(make_w2_kill_profiles.twin_names()))
