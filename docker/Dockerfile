#syntax=docker/dockerfile:1.6
# Thin submission image: weights are NOT baked in — they are uploaded
# separately as weights.zip and mounted at /weights at runtime
# (same flow as the reportgen_example_docker baseline).
#
# Build:  ./build.sh
# Test:   ./test.sh       (needs ./weights/ populated — see scripts/hf_download_weights.sh)
# Export: ./export.sh

# Base choice, driven by the storm build host's Docker 19.03:
#   * ubuntu20.04 (focal, glibc 2.31): Ubuntu 22.04+ images need Docker
#     >= 20.10.10 (old runc's seccomp answers jammy-glibc's faccessat2 with
#     EPERM, so even root "cannot read" the apt keyrings). Focal predates
#     faccessat2 and builds fine on 19.03.
#   * -base- (not -cudnn-runtime-): provides libcudart per the challenge
#     checklist, while torch's cu126 pip wheels bundle their own cuDNN/cuBLAS
#     — the runtime variant would duplicate ~3 GB against the 15 GB image cap.
FROM nvidia/cuda:12.6.3-base-ubuntu20.04

# -------- system deps --------
# python3.9 (focal universe): newest interpreter available without a PPA;
#   transformers 4.57 needs >= 3.9 and the whole pipeline source parses on 3.9
#   (all `X | Y` annotations sit behind `from __future__ import annotations`).
# pip self-upgrade is required: focal's pip 20.0 does not know the
#   manylinux_2_28 tag that torch 2.8 cu126 wheels use.
# rm cuda*.list: the NVIDIA apt repo is unused here and its HTTPS endpoint
#   fails behind TLS-inspecting proxies — drop it instead of trusting it.
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3.9-distutils python3-pip git ca-certificates \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade \
        --trusted-host pypi.org --trusted-host files.pythonhosted.org \
        pip setuptools wheel

WORKDIR /opt/app

# Build hosts behind TLS-inspecting proxies (e.g. the NIH-managed build box)
# fail pip's certificate verification; trusted-host skips it for the pinned
# indexes only. Build-time only — the submission runs fully offline.
# download-r2.pytorch.org: the pytorch index redirects wheel downloads there.
# pypi.nvidia.com: serves torch's nvidia-*-cu12 dependency wheels.
RUN printf '[global]\ntrusted-host =\n    pypi.org\n    files.pythonhosted.org\n    download.pytorch.org\n    download-r2.pytorch.org\n    pypi.nvidia.com\n' > /etc/pip.conf

# General fix for the TLS-inspecting proxy: build.sh snapshots the build
# host's trusted CA bundle (which includes the proxy's CA) into the context;
# if present, point pip at it so certificate verification succeeds for ANY
# host instead of playing trusted-host whack-a-mole. Public certs only.
COPY host-ca-bundle.crt /usr/local/share/host-ca-bundle.crt
RUN if [ -s /usr/local/share/host-ca-bundle.crt ]; then \
        printf 'cert = /usr/local/share/host-ca-bundle.crt\n' >> /etc/pip.conf; \
    fi

# -------- python deps (CUDA-matched torch first, then the pinned set) --------
# torch 2.8.0 matches the PETRG-3D training/inference venv; cu126 wheels run on
# both L4 (sm_89) and A100 (sm_80).
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# -------- rad-vision-engine (rve + vision_engine) --------
COPY src/vendor/rad-vision-engine/ /opt/app/rad-vision-engine/
RUN pip install --no-cache-dir --no-deps -e /opt/app/rad-vision-engine/ && \
    rm -rf /root/.cache/pip

# -------- pipeline sources --------
COPY src/vendor/petrg/           ./src/vendor/petrg/
COPY src/vendor/pillar_min/      ./src/vendor/pillar_min/
# pillar_backbone = pillar-finetune's port of the lapa tower. Stage A's
# cls probe and retrieval head were trained on ITS numerics, not on the petrg
# port that CTRG-3D carries; both towers ship on purpose (see visual_cache.py).
COPY src/vendor/pillar_backbone/ ./src/vendor/pillar_backbone/
COPY src/pipeline/               ./src/pipeline/
COPY assets/                ./assets/
COPY predict.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# -------- runtime env --------
# NUM_VISUAL_TOKENS must match the CTRG-3D checkpoint packed into weights.zip.
# MAX_SEQ is 2688, not 3072: that is generate_batched.py's default, and
# run_sbatch_grpo_score_newencB never overrides it, so it is the truncation
# length every ranked draw was produced with.
# GEN_BATCH 24 measured 29.4 GB on an A100-40.
#
# -------- eleventh draw: text-only --------
# Stock Qwen3-8B + a LoRA trained on cls/ret templates alone (no vision). A
# different model class, so its errors do not correlate with the ten CTRG-3D
# draws. Measured on valid 1,552 as the 11th voter: macro-F1 0.6123 -> 0.6126,
# the highest of any configuration.
#   TEXTONLY_ADAPTER  textonly_ep8  = checkpoint-11928, generated at cap 2048
#                     textonly_ep11 = checkpoint-16401, generated at cap 768
#   Both ship (15 MB each). ep8 (TO11k3F5I0, sum 1.5571) is 0.0005 ahead of
#   ep11 (T11k3F5I0, 1.5566) -- 1/16 of the noise floor, i.e. indistinguishable.
#   ep11 SHIPS because it is the one the 11-draw replay proved: stage-D output
#   == T11k3F5I0.csv on 300/300 exams (job 28703412). Do not switch to ep8
#   without its replay passing; the sum difference cannot justify shipping an
#   unverified draw. NOTE ep11's scored draw was generated at cap 768, so
#   TEXTONLY_MAX_NEW must be 768 for it and 2048 for ep8.
# VLOOP_ITERS=0: with a text-only draw in the pool the verify loop donates its
# prose and loses ground (bbTk3 1.5571 -> bbTk3_v 1.5533). The ten-draw
# submission keeps the loop; this one must not.
ARG NUM_VISUAL_TOKENS=128
# GTD5I0_v: nine CTRG-3D draws + the text-only draw spliced in at slot 5,
# skeleton at slot 6. BACKBONE_IDX is stated against the FINAL ten-slot list and
# is NOT auto-shifted, so it must be given explicitly whenever
# TEXTONLY_INDEX <= BACKBONE_IDX. N_DRAWS is the CTRG-3D count (nine); the DRAWS
# literal in predict.py is checked against it at import.
ENV N_DRAWS=9 \
    BACKBONE_IDX=6 \
    TEXTONLY_DRAW=1 \
    TEXTONLY_INDEX=5 \
    TEXTONLY_ADAPTER=textonly_lam05_200 \
    TEXTONLY_MAX_NEW=768 \
    TEXTONLY_MAX_SEQ=2432 \
    TEXTONLY_BATCH=16 \
    MIN_SUPPORT=3 \
    MERGE_MAX_ADD=5 \
    MERGE_MAX_ADD_IMP=-1 \
    VLOOP_ITERS=2
ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    CHECKPOINT_DIR=/checkpoint \
    WEIGHTS_DIR=/weights \
    NUM_VISUAL_TOKENS=${NUM_VISUAL_TOKENS} \
    MAX_SEQ=2688 \
    GEN_BATCH=24 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1

# Non-root user (validator rejects root).
# /checkpoint and /output are pre-created runner-owned: locally-created named
# volumes inherit the image dir's ownership on first (empty) mount, and uid
# 1000 cannot mkdir at / when the platform doesn't mount them.
RUN useradd --create-home --uid 1000 runner && \
    mkdir -p /checkpoint /output && \
    chown -R runner:runner /opt/app /checkpoint /output
USER runner

ENTRYPOINT ["/opt/app/entrypoint.sh"]
