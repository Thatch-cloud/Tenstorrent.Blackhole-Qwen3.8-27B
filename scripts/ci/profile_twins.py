"""Every generated profile twin the profile-enumerating tests exempt (each generator's own test holds its twins): engine reuse's (make_parked_profiles),
round-host's (make_round_host_profiles), octo-T8's (make_octo_profiles), the tp4/octo-2 levers' (make_octo2_profiles), the tp4/octo-3 twins of the fused stack's (make_octo3_profiles), the tp4/ship-4 twins of the combined ship candidate (make_ship4_profiles), the tp4/np1-fastlane twins of the approved ship-4 profile (make_np1_profiles), the tp4/next-1 audited twin of the ship-5 candidate (make_s5_profiles), the tp4/exact-end twins of the approved ship-4 profile (make_xe_profiles), the tp4/prefill-2 twins of the approved ship-4 profile (make_pf2_profiles), the tp4/prefill-3 K-split twins of the ship-5 candidate (make_pf3k_profiles), the tp4/sched-1 twins of the approved ship-4 profile (make_sched_profiles), the W2 kill drill's (make_w2_kill_profiles), the
region-read audit's (make_kvread_profiles), the host KV tier's (make_prefix_tier_profiles), the engine-start upload levers' (make_upload_profiles) and the op-fusion
programme's (make_fusion_profiles, from the manifests in fusion-wp). One list, so a lever branch that adds twins adds them here and not to a dozen tests."""

import make_fusion_profiles
import make_kvread_profiles
import make_np1_profiles
import make_xe_profiles
import make_octo2_profiles
import make_octo3_profiles
import make_octo_profiles
import make_pf2_profiles
import make_pf3k_profiles
import make_parked_profiles
import make_prefix_tier_profiles
import make_round_host_profiles
import make_s5_profiles
import make_sched_profiles
import make_ship4_profiles
import make_upload_profiles
import make_w2_kill_profiles


def twin_names():
    return (tuple(make_parked_profiles.twin_names()) + tuple(make_round_host_profiles.twin_names())
            + tuple(make_octo_profiles.twin_names()) + tuple(make_octo2_profiles.twin_names()) + tuple(make_octo3_profiles.twin_names()) + tuple(make_ship4_profiles.twin_names())
            + tuple(make_np1_profiles.twin_names()) + tuple(make_s5_profiles.twin_names())
            + tuple(make_xe_profiles.twin_names())
            + tuple(make_pf2_profiles.twin_names())
            + tuple(make_pf3k_profiles.twin_names())
            + tuple(make_sched_profiles.twin_names())
            + tuple(make_w2_kill_profiles.twin_names()) + tuple(make_kvread_profiles.twin_names())
            + tuple(make_prefix_tier_profiles.twin_names())
            + tuple(make_upload_profiles.twin_names()) + tuple(make_fusion_profiles.twin_names()))


def generated_names():
    """twin_names plus the TRAFFIC profiles an integration generator writes (the op-fusion ship candidate and its tier variant: not gate-only, so not twins). The census tests that enumerate the traffic profiles
    exempt these; the tests that assert a property of the gate-only twins use twin_names."""
    return twin_names() + tuple(make_fusion_profiles.ship_names()) + tuple(make_prefix_tier_profiles.ship_names())
