#!/usr/bin/env bash

prepare_dflash_fixtures() {
    dflash_cache=/home/thatch/.cache/qwen-experiments
    dflash_revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
    for component in attention convolution mlp projection selector; do
        fetcher="draft_${component}_fixture.py"
        if [ "$component" = projection ]; then fetcher=draft_projection_full_fixture.py; fi
        fixture="$dflash_cache/dflash2-$component-$dflash_revision"
        timeout -k 10 1200 python3 "scripts/ci/$fetcher" --reuse-verified --output "$fixture"
        cp "$fixture/manifest.json" "$output/dflash-$component-manifest.json"
    done
    timeout -k 10 4800 python3 -u scripts/ci/draft_remaining_layers_fixture.py --reuse-verified \
        --output "$dflash_cache/dflash2-stack-$dflash_revision"
    for layer in 1 2 3 4; do
        cp "$dflash_cache/dflash2-stack-$dflash_revision/layer-$layer/manifest.json" "$output/dflash-layer-$layer-manifest.json"
    done
    dflash_layout="$output/dflash-fixture-layout"
    mkdir -p "$dflash_layout"/{attention,convolution,mlp,projection,selector,layer-1,layer-2,layer-3,layer-4}
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
