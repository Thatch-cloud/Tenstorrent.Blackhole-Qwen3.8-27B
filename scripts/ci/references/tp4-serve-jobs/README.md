# Job files for serving general-prefix-tp4 on the four-card rig

Templates for `.github/c2-serving-job.env`, one per step of the four-card G1 route (build, reset + prefix gate, replay,
status, verify, unserve, re-serve reset). Each file's header says when to run it and what to read.

Two placeholders are filled when a job is pushed:

- `C1`: the lowercase short sha of the commit whose tree built the image (`C2_IMAGE_TAG=tp4-g1-C1`).
- `TS7`: the Thatch.Server short sha in the thin layer's tag (`C2_PLATFORM_IMAGE=...thatch-serving-tt:TS7`).

Copy a template over `.github/c2-serving-job.env` on a job commit, write it with LF line endings, commit, move the
`experiment/c2-serving-vN` tag and push it. `scripts/ci/test_tp4_serve_jobs.py` parses every template here under the
checkout's own job parser, so a template cannot drift from what the workflow accepts.
