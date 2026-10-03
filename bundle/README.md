# bundle/: the serving runtime tree for stage 7

`docker/two-card/qwen-fast-serving.Dockerfile` (stage 7) copies four things from a `bundle/` directory in its build
context. This is that directory: a flattened, sanitised copy of the runtime tree the qualified P8 serving image
carries, so stage 7 can be built from a clone of this repository.

| Here | Lands in the image at |
|---|---|
| `bundle/experiment-scripts/` | `/experiment-scripts` |
| `bundle/experiment-optimisation/` | `/experiment-optimisation` (the Dockerfile links `/optimisation` to it) |
| `bundle/speculative-decoding/` | `/speculative-decoding` |
| `bundle/serving-bundle.json` | `/opt/qwen-serving/serving-bundle.json` |

Build from the repository root:

```bash
docker build -f docker/two-card/qwen-fast-serving.Dockerfile \
  -t localhost:5000/qwen-fast-serving:p8 .
```

## What it is, exactly

- **Pristine bundle, not the overlay.** The files are the original bundle. The files that stage 7 overwrites from
  `scripts/ci/` (over a hundred: `serving_runtime.py`, `serving_worker_hook.py`, `model_batch.py`, the draft branches and
  so on) are not repeated here; the Dockerfile's `COPY scripts/ci/...` lines supply them. If a name is in both places,
  `scripts/ci/` wins, as in the image.
- **Many files here are also in `scripts/ci/`.** Where the bundle's copy and the tracked copy are byte-identical the
  bundle still carries it, so that `/experiment-scripts/ci` is complete without relying on which files the Dockerfile
  happens to name.
- **No binaries and no bytecode.** The image's `_ttnncpp.so` is not here (stage 6 builds yours), and there are no
  `.so` or `__pycache__` files.
- **Line endings.** Every file is LF with no carriage returns, so `.gitattributes` leaves the bytes alone and the
  sha256 pins inside the evidence files still hold after a commit and checkout. Do not re-save these files with
  another editor's line-ending settings.

## Integrity

`serving-bundle.json` is a manifest of this directory: a `sources` map of path (relative to `bundle/`) to sha256, for
every file except `serving-bundle.json` and this README. Check it with:

```bash
python3 - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path('bundle')
sources = json.loads((root / 'serving-bundle.json').read_text())['sources']
bad = [p for p, h in sources.items() if hashlib.sha256((root / p).read_bytes()).hexdigest() != h]
print(len(sources), 'files,', len(bad), 'mismatches', bad[:5])
EOF
```

Nothing at run time reads this file; it is here because the Dockerfile copies it and so that the tree can be verified.
(The maintainers' original `serving-bundle.json` described the unflattened tree, which no longer matches; it is replaced.)

## Byte-exact files: do not edit

The speculative-decoding profiles (`exact`, `c2`, `c2-packed`) read evidence records that pin the sha256 of the files
they were measured against. These are published unchanged and must stay unchanged: `frozen-evidence/` (the report
files and the copies they pin), `dflash-t16-native-evidence/`, the `*-evidence/` directories' `.json` and
`.exit-status` records, `frozen-gdn-norm.json`, `shared-qk-norm-scatter.json`, `frozen-draft-tail-hardware.json`,
`history-append-hardware.json`. The `general*` profiles read none of them.

The evidence records do pin native tt-metal sources that your build may not reproduce; BUILD.md, "Pins that `exact`,
`c2` and `c2-packed` check", lists which and what to do.

## What is withheld, and what that costs

Files were left out when they carried operator-specific material (host paths, registry hosts, runner names, card
serials, image digests, CI run identifiers) and nothing the serving path loads needs them:

- host-side driver scripts for the maintainers' own rig (`run-baseline.sh`, `run-simulator.sh`, `run-hardware.sh`,
  `run-dspark-hardware.sh`, `reset-cards.sh`, the inventory and card-owner helpers) and their tests;
- recorded runner artifacts (the `simulator-assets.sha256`, `runner-io-admission.json` and `cache.patch` files in the
  `*-evidence/` directories, the `dspark-*-hardware*.json` run logs, the `dspark-*-runtime-cache.json` records);
- the 28 files of `frozen-evidence/{draft,target}/scripts/ci/` that no report pins (the gate reads only the pinned ones);
- three probe scripts that quote CI run identifiers (`full-prefix.py`, `attention-replay.py`, `gdn-vsplit-timing.py`);
- the native-cache manifest of the original build and the install records `native-install.json` and
  `startup-preflight.json` (build outputs; stage 7 writes `startup-preflight.json` itself).

Consequences you may hit: the simulator/evidence-regeneration drivers (anything that re-runs the maintainers' rig
gates) are not runnable from this tree, and some evidence records list source paths that are not published here (they
record what the maintainers' tree held). A static trace of the modules stage 7 loads found no code that re-hashes those listed paths.

One file was edited: `experiment-scripts/ci/frozen_recipe_context.py` (a volume name, a container label prefix and a
comment). No evidence record pins it.

Provenance: cut from the P8 serving image at the layers the Dockerfile writes, minus the above. The tree is not itself
a qualified artifact: both flags in `serving-bundle.json` are `false`.
