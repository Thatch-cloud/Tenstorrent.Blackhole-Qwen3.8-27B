#!/usr/bin/env bash

prepare_dflash_fixtures() {
    dflash_cache=/home/thatch/.cache/qwen-experiments
    dflash_revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
    # C2_DRAFTER_MANIFEST names a candidate drafter (references/drafter-manifests): its fixtures are staged beforehand by
    # drafter_stage.py (no fetch here), and the loader checks every byte against that manifest at attach.
    dflash_manifest=${C2_DRAFTER_MANIFEST:-dedf8df6}
    if [ "$dflash_manifest" != dedf8df6 ]; then
        dflash_revision=$(python3 -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["revision"])' \
            "scripts/ci/references/drafter-manifests/$dflash_manifest.json")
    fi
    for component in attention convolution mlp projection selector; do
        fetcher="draft_${component}_fixture.py"
        if [ "$component" = projection ]; then fetcher=draft_projection_full_fixture.py; fi
        fixture="$dflash_cache/dflash2-$component-$dflash_revision"
        if [ "$dflash_manifest" = dedf8df6 ]; then
            timeout -k 10 1200 python3 "scripts/ci/$fetcher" --reuse-verified --output "$fixture"
        else
            test -f "$fixture/manifest.json"
        fi
        cp "$fixture/manifest.json" "$output/dflash-$component-manifest.json"
    done
    if [ "$dflash_manifest" = dedf8df6 ]; then
        timeout -k 10 4800 python3 -u scripts/ci/draft_remaining_layers_fixture.py --reuse-verified \
            --output "$dflash_cache/dflash2-stack-$dflash_revision"
    else
        test -f "$dflash_cache/dflash2-stack-$dflash_revision/layer-4/manifest.json"
    fi
    for layer in 1 2 3 4; do
        cp "$dflash_cache/dflash2-stack-$dflash_revision/layer-$layer/manifest.json" "$output/dflash-layer-$layer-manifest.json"
    done
    dflash_layout="$output/dflash-fixture-layout"
    mkdir -p "$dflash_layout"/{attention,convolution,mlp,projection,selector,layer-1,layer-2,layer-3,layer-4}
    if [ "$dflash_manifest" != dedf8df6 ]; then printf '%s\n' "$dflash_manifest" > "$dflash_layout/DRAFTER_MANIFEST"; fi
}

copy_dflash_fixtures() {
    docker cp "$dflash_layout" "$test_id:/experiment-dflash-fixture"
    for component in attention convolution mlp projection selector; do
        docker cp "$dflash_cache/dflash2-$component-$dflash_revision/." "$test_id:/experiment-dflash-fixture/$component"
    done
    for layer in 1 2 3 4; do
        docker cp "$dflash_cache/dflash2-stack-$dflash_revision/layer-$layer/." "$test_id:/experiment-dflash-fixture/layer-$layer"
    done
}
