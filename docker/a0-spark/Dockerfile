# The A0 screen's GPU image: torch + transformers + the fast GDN kernels, and the screen's own scripts. Nothing here pulls a model:
# weights, the bundle and the pins are mounted at run time and checked against their sha256 first (a0_upstream.py).
#
#   docker build --build-arg BASE=<an aarch64 CUDA PyTorch base image the operator already holds> \
#       --build-arg TRANSFORMERS_VERSION=<pin> --build-arg FLA_VERSION=<pin> -f docker/a0-spark/Dockerfile -t a0-screen .
#
# The base image is a build ARG on purpose: its name belongs in the operator's private run plan, not in this public repository.
ARG BASE
FROM ${BASE}

ARG TRANSFORMERS_VERSION
ARG FLA_VERSION
RUN test -n "${TRANSFORMERS_VERSION}" && test -n "${FLA_VERSION}" \
 && python3 -m pip install --no-cache-dir "transformers==${TRANSFORMERS_VERSION}" "flash-linear-attention==${FLA_VERSION}" safetensors

WORKDIR /a0
COPY scripts/ci/a0_run.py scripts/ci/a0_bundle_io.py scripts/ci/a0_target.py scripts/ci/a0_drafters.py scripts/ci/a0_upstream.py      scripts/ci/dflash2_torch.py scripts/ci/tf_pair_walk.py scripts/ci/tf_pair_report.py scripts/ci/tau_lab_report.py      scripts/ci/acceptance_report.py scripts/ci/extent_attention_replay.py /a0/
# a0_watchdog.py is NOT in the image: it runs on the host beside the container (it needs the docker CLI) and is started first.

# the screen runs as an unprivileged user with its working files private
RUN useradd --create-home --uid 1000 a0 && chown -R a0 /a0
USER a0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/a0
ENTRYPOINT ["python3", "/a0/a0_run.py"]
