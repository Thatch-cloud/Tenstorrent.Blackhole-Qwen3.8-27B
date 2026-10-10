"""Every generated profile twin the profile-enumerating tests exempt (each generator's own test holds its twins): engine reuse's (make_parked_profiles),
round-host's (make_round_host_profiles), octo-T8's (make_octo_profiles), the tp4/octo-2 levers' (make_octo2_profiles), the W2 kill drill's (make_w2_kill_profiles) and the
region-read audit's (make_kvread_profiles), the engine-start upload levers' (make_upload_profiles) and the op-fusion programme's (make_fusion_profiles, from the manifests in fusion-wp). One list, so a lever branch that adds twins adds them here and not to a dozen tests."""

import make_fusion_profiles
import make_kvread_profiles
import make_octo2_profiles
import make_octo_profiles
import make_parked_profiles
import make_round_host_profiles
import make_upload_profiles
import make_w2_kill_profiles


def twin_names():
    return (tuple(make_parked_profiles.twin_names()) + tuple(make_round_host_profiles.twin_names())
            + tuple(make_octo_profiles.twin_names()) + tuple(make_octo2_profiles.twin_names())
            + tuple(make_w2_kill_profiles.twin_names()) + tuple(make_kvread_profiles.twin_names())
            + tuple(make_upload_profiles.twin_names()) + tuple(make_fusion_profiles.twin_names()))


def generated_names():
    """twin_names plus the TRAFFIC profiles an integration generator writes (the op-fusion ship candidate: not gate-only, so not a twin). The census tests that enumerate the traffic profiles
    exempt these; the tests that assert a property of the gate-only twins use twin_names."""
    return twin_names() + tuple(make_fusion_profiles.ship_names())
