# fusion-wp manifests

One file per work package, `scripts/ci/fusion-wp/<WP>.json` (WP1, WP2, WP5, WP6, WP7). A package never edits the shared files
(profiles, smoke dispatch, overlay manifest, CPU allowlist, tp_addresses); `python3 scripts/ci/make_fusion_profiles.py --write`
places the manifest's entries there, inside fenced blocks. `--report` prints where every entry landed; `--check` fails when a
block is stale. A manifest with a key the generator cannot place is refused by name.

```json
{
  "wp": "WP1",
  "branch": "tp4/fx-wp1",
  "levers": [
    {
      "id": "s1",
      "name": "S1 shard argmax",
      "flag": "QWEN_FAST_TP4_SHARD_ARGMAX",
      "value": "1",
      "audit_flag": "QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT",
      "marker": "tp4 shard argmax",
      "env": {},
      "audit_env": {},
      "reason": "one line: what the lever fuses"
    }
  ],
  "smoke_rules": ["tp4_shard_argmax_smoke"],
  "image_files": [{"path": "scripts/ci/tp4_shard_argmax_fold2.cpp", "reason": "JIT source of the lever"}],
  "tests": ["test_tp4_shard_argmax_fold2", {"discover": "optimisation/ttnn-op/shard_argmax"}],
  "tp_addresses": {"module_twins": [["quad_draft", "quad_draft_tp"]]},
  "notes": "free text, ignored"
}
```

## What each key becomes

| key | lands in | rule |
| --- | --- | --- |
| `levers[]` | `qwen_c2_profiles.json`, `c2_smoke_check.py` | per lever two gate-only twins of the production profile, `<production>-fx-<id>` (the flag on: the timed arm) and `...-fx-<id>-audit` (plus `audit_flag=1`); the smoke tables (engaged line required when the profile sets the flag, a fell-back line always fails, an audit flag needs a line with `exact=True`) |
| `profiles[]` | `qwen_c2_profiles.json` | an extra twin: `{"name": "<id>", "env": {...}, "reason": "...", "audit": false}`; `name` is the suffix after `-fx-`; `parent` defaults to the production profile |
| `smoke[]` | `c2_smoke_check.py` | a marker row without a profile: `{"flag", "value", "marker" or "engaged"/"fell_back"/"audit", "audit_flag", "what"}` |
| `smoke_rules[]` | `c2_smoke_check.py` | the package's own stricter smoke rule: a host-side module of `scripts/ci` (`"tp4_shard_argmax_smoke"`) with `problems(env, container_text) -> [problem]`, called once per arm by `fusion_problems` (not an image file) |
| `pack_arms[]` | `references/fusion-jobs` | extra profile twins (`profiles[]`) that join the integrated card pack: the suffixes (`"s1f2"`); a lever is in the pack already, and an extra twin that no manifest names stays with its package's own folder |
| `pack_order[]` | `references/fusion-jobs` | lever ids or pack-arm suffixes in the order their audits and ABABs run (the unlisted follow in manifest order, `pack_last` at the end): the integrator's priority, by the plan's saving per round |
| `pack_last[]` | `references/fusion-jobs` | lever ids or pack-arm suffixes (or `{"arm", "reason"}`) whose audited attach and ABAB run after every other arm in the integrated ORDER, the reason named in its header: a lever that cannot engage until a card-M record lands |
| `image_files[]` | `docker/qwen-c2-overlay.txt` | strings or `{"path", "reason"}`, under `scripts/ci/` (every new module, kernel source, smoke module and any module an overlaid file imports); the overlay is the ONE image list, the P8 copy lists are not touched |
| `tests[]` | the CPU allowlist (the regression step the any-ref CPU suite runs) | module names (`test_x`, or `test_x.Class.method`) or `{"discover": "<dir>", "pattern": "test_*.py"}` |
| `tp_addresses` | `tp_addresses.py` | `twins` (module, attr, twin module, twin attr), `module_twins` (module, twin module), `flagged_twins` / `flagged_module_twins` (`{"module", "attr", "flag", "gate": "module:function"}`, the gate takes the environment mapping) |

`marker: "tp4 shard argmax"` means the three prefixes `[PINDIAG] tp4 shard argmax engaged`, `... fell back` and `... audit`
(the audit line must also carry `exact=True` when it passes). Name them one by one with `engaged`, `fell_back`, `audit` if the
words differ. A flag the static smoke tables already dispatch is recognised and not repeated.

Refused: an unknown key; a file named in `image_files`, `tests` or `tp_addresses` that is not in the tree; a lever flag already in
the parent profile; two packages naming one lever id, flag or profile name; an overlay line `c2_overlay.py` refuses (a pinned file,
a duplicate, a path outside `scripts/ci/`).

## Card jobs

A package's own templates live in `scripts/ci/references/fusion-jobs/<WP>/` (its card-M jobs, its detail jobs, its read and decision rules; every template passes
`python3 -s scripts/ci/c2_serving_job.py <env> scripts/ci/qwen_c2_profiles.json` with rc 0, naming the one image `tp4-fusion-1`). The files directly in `fusion-jobs/` are the
integrator's: `make_fusion_jobs.py` writes B0, X0, Z and, for every lever (and every extra profile twin named under `pack_arms`), an audited attach (`<ID>A`) and a timed control/lever ABAB at eight
live (`<ID>C1 <ID>L1 <ID>C2 <ID>L2`); an extra twin `x` with an audit twin `x-audit` is one arm. A package may keep its templates in
`references/fusion-jobs/<WP>/` or in a sibling `references/fusion-*` folder.

`WP0.json` is the integrator's own manifest (same schema): wiring a package asked for in its notes (for example a `smoke_rules` entry for its stricter smoke module), kept out of the
package's file so that the package's manifest stays exactly as it pushed it.
