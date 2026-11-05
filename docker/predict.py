#!/usr/bin/env python
"""VLM3D report-generation submission — CTRG-3D newenc-B + GRPO 10-draw ensemble.

Reproduces `bb1000F5I0_v`, the top-ranked case of PETRG-3D's 246-way comparison
on the full valid split (CRG 0.5749 / macro-F1 0.6123 / BLEU_mean 0.37232).

PIPELINE, PER RUN
=================
  Stage A   (two vision towers resident, ~1.5 GB)
    For every /input/*.nii.gz, once:
      1. RAVE chest-CT preprocessing (1.25 mm iso, middle 256 slices, 256x256).
      2. pillar-finetune's lapa tower -> {"pooled" 384, "activ" (384,16,8,8)}
         -> cls template (per-class F1-max thresholds, 18 sentences)
         -> ret template (Simple_TranSQ + last-token gallery, thr 0.4, + `_pp`)
      3. PETRG-3D's lapa tower + Perceiver + fc -> (128, 4096) visual
         embedding, appended to a float16 memmap.
    Two towers because the heads were trained on different ports of the same
    encoder; see pipeline/visual_cache.py. One volume read either way.

  Stage B   (8B LLM resident; stage A freed first)
    Load CTRG-3D once. For each of 10 draws: overwrite the 144 `lora_*` tensors
    (15 MB) and generate over every volume in batches, reading vision from the
    memmap. No volume is touched again and no model is reloaded.

  Stage C   RadBERT: 18 labels per draw per exam (the scorer's own path), plus
            labels for the pooled candidate sentences.
  Stage D   Label-guided merge: vote k=3 of 10, skeleton = draw 6, repair.
  Stage E   Read-back verify loop, 2 iterations, <=2 added sentences per exam.
            -> GC-wrapped /output/predictions.json

Idempotent: every stage checkpoints into /checkpoint, so a SIGTERM resumes
without recomputation. Per-volume inference runs WITHOUT try/except on purpose:
a failing volume must crash the run loudly rather than silently ship partial
predictions.

Environment variables (all optional):
  INPUT_DIR / OUTPUT_DIR / CHECKPOINT_DIR / WEIGHTS_DIR
  NUM_VISUAL_TOKENS  perceiver latents; MUST match the CTRG-3D checkpoint (128)
  MAX_SEQ            prompt truncation (default 2688 — generate_batched.py's)
  GEN_BATCH          stage B batch size (default 24; 29.4 GB measured on A100-40)
  RET_THRESHOLD      retrieval firing threshold (default 0.4)
  N_DRAWS            how many of the 10 draws to run (default 10; lowering it
                     changes the recipe — see MIN_SUPPORT below)
  MIN_SUPPORT        vote threshold (default 3)
  MERGE_MAX_ADD / MERGE_MAX_ADD_IMP / VLOOP_ITERS / VLOOP_MAX_ADD
"""
from __future__ import annotations

import json
import contextlib
import os
import re
import signal
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------- paths & knobs ----------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))                       # pipeline
sys.path.insert(0, str(ROOT / "src" / "vendor" / "petrg"))  # Model, Dataset
sys.path.insert(0, str(ROOT / "src" / "vendor"))            # pillar_min, pillar_backbone

INPUT_DIR      = Path(os.environ.get("INPUT_DIR",      "/input"))
OUTPUT_DIR     = Path(os.environ.get("OUTPUT_DIR",     "/output"))
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoint"))
WEIGHTS_DIR    = Path(os.environ.get("WEIGHTS_DIR",    "/weights"))

NUM_VISUAL_TOKENS = int(os.environ.get("NUM_VISUAL_TOKENS", "128"))
MAX_SEQ           = int(os.environ.get("MAX_SEQ", "2688"))
GEN_BATCH         = int(os.environ.get("GEN_BATCH", "24"))
RET_THRESHOLD     = float(os.environ.get("RET_THRESHOLD", "0.4"))
SEED              = int(os.environ.get("SEED", "42"))

N_CTRG_DRAWS      = int(os.environ.get("N_DRAWS", "10"))
MIN_SUPPORT       = int(os.environ.get("MIN_SUPPORT", "3"))

# Eleventh draw: stock Qwen3-8B + text-only LoRA, no visual input. A different
# model class, so its errors do not correlate with the ten CTRG-3D draws.
# Measured on valid 1,552: macro-F1 0.6123 -> 0.6126 (the highest of any config).
TEXTONLY_DRAW     = os.environ.get("TEXTONLY_DRAW", "0") == "1"
TEXTONLY_ADAPTER  = os.environ.get("TEXTONLY_ADAPTER", "textonly_lora")
TEXTONLY_MAX_NEW  = int(os.environ.get("TEXTONLY_MAX_NEW", "2048"))
TEXTONLY_MAX_SEQ  = int(os.environ.get("TEXTONLY_MAX_SEQ", "2432"))
TEXTONLY_BATCH    = int(os.environ.get("TEXTONLY_BATCH", "16"))
# Total draws the vote sees. The text-only draw is always last so that
# BACKBONE_IDX keeps indexing the CTRG-3D list.
N_DRAWS           = N_CTRG_DRAWS + (1 if TEXTONLY_DRAW else 0)
MERGE_MAX_ADD     = int(os.environ.get("MERGE_MAX_ADD", "5"))
MERGE_MAX_ADD_IMP = int(os.environ.get("MERGE_MAX_ADD_IMP", "-1"))
VLOOP_ITERS       = int(os.environ.get("VLOOP_ITERS", "2"))
VLOOP_MAX_ADD     = int(os.environ.get("VLOOP_MAX_ADD", "2"))
# How often stage A rewrites state.json. It is a whole-file rewrite whose
# size grows with progress, so doing it per volume is 2,000 rewrites of an
# object that reaches ~8 MB. A crash costs at most this many redone volumes.
CKPT_EVERY        = int(os.environ.get("CKPT_EVERY", "25"))

# Stage-A I/O staging + per-step profiling (see _staged_input / _Prof).
INPUT_STAGE = os.environ.get("INPUT_STAGE", "1") == "1"
STAGE_DIR   = Path(os.environ.get("STAGE_DIR", "/tmp/stage_in"))
QUIET_RVE   = os.environ.get("QUIET_RVE", "1") == "1"

HIDDEN = 4096

STATE_PATH      = CHECKPOINT_DIR / "state.json"
WORK_DIR        = CHECKPOINT_DIR / "work"
VOLUME_DIR      = WORK_DIR / "volumes"
VISUAL_PATH     = WORK_DIR / "visual.npy"
VISUAL_DIR      = WORK_DIR / "visual"      # one 1 MB .npy per volume
DRAW_DIR        = WORK_DIR / "draws"
LABEL_DIR       = WORK_DIR / "labels"
PREDICTIONS_OUT = OUTPUT_DIR / "predictions.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# The ensemble, exactly as ranked
# --------------------------------------------------------------------------
# Ten draws over seven distinct LoRA adapters -- draws 2..4 re-sample the SAME
# adapter with different sampler settings, which is why only seven adapter
# slices ship. Order is load-bearing: BACKBONE_IDX indexes into it, and the
# skeleton draw is what supplies the prose the merge keeps.
#
# Sampler defaults (test.py's, verbatim): do_sample, T 0.7, top_p 0.9,
# repetition_penalty 1.05, max_new_tokens 2048, seed 42.
DEFAULT_SAMPLER = dict(temperature=0.7, top_p=0.9, repetition_penalty=1.05,
                       max_new_tokens=2048)

DRAWS = [
    # GBF5I0_v -- top-ranked pair with bb1000F5I0_v's plumbing untouched.
    # Same recipe as before (k=3, backbone index 6, Findings unlimited,
    # Impression 0, verify loop 2); four draw members differ. No text-only draw
    # and no per-draw seed, so nothing here needs new machinery.
    #
    # ORDER IS LOAD-BEARING beyond BACKBONE_IDX: label_guided_merge's
    # cluster_block() keeps, as each cluster's representative, the sentence of
    # the FIRST draw that opened it -- and that representative is what ships in
    # the report. Reordering produces different prose, not just a different
    # tie-break, so this must match the scored run line for line.
    ("grpo_newencB-700",             "grpo_newencB_700",      {}),
    ("grpo_newencB-800",             "grpo_newencB_800",      {}),
    ("grpo_B_a2from700-1000_t10p95", "grpo_B_a2from700_1000", {"temperature": 1.0, "top_p": 0.95}),
    ("grpo_newencB-900_t10p95",      "grpo_newencB_900",      {"temperature": 1.0, "top_p": 0.95}),
    ("grpo_newencB-800_rep100_t10p95", "grpo_newencB_800",    {"repetition_penalty": 1.0, "temperature": 1.0, "top_p": 0.95}),
    ("grpo_B_g3-800",                "grpo_B_g3_800",         {}),
    ("grpo_B_a2from700-1000",        "grpo_B_a2from700_1000", {}),   # <- skeleton
    ("grpo_newencB-1100_rep100",     "grpo_newencB_1100",     {"repetition_penalty": 1.0}),
    ("grpo_B_nobleu-700",            "grpo_B_nobleu_700",     {}),
    ("grpo_B_g3-700",                "grpo_B_g3_700",         {}),
]
BACKBONE_IDX = 6


# ---------- signal handling ----------
_stop_requested = False


def _on_sigterm(signum, frame):
    global _stop_requested
    print("[predict] SIGTERM received; will flush state and stop at the next "
          "checkpoint", file=sys.stderr, flush=True)
    _stop_requested = True


signal.signal(signal.SIGTERM, _on_sigterm)


# ---------- filename -> input_image_name (same rule as the baseline docker) ----
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def derive_image_name(nii_path: Path) -> str:
    stem = nii_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    for p in reversed(stem.split("_")):
        if UUID_RE.fullmatch(p):
            return p
    return stem


# ---------- state ----------
def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception as e:
            print(f"[predict] unreadable checkpoint state ({e}); starting fresh",
                  file=sys.stderr)
    return {"stage_a_done": [], "cls_entries": {}, "ret_entries": {},
            "draws_done": {}, "stage_c_done": False}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    os.replace(tmp, STATE_PATH)


def write_final_predictions(reports: dict, names: list[str]) -> None:
    """Grand-challenge-wrapped schema consumed by vlm3d-reportgen-eval."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inner = {
        "name": "Generated reports",
        "type": "Report generation",
        "version": "1.0",
        "generated_reports": [{"input_image_name": n, "report": reports[n]}
                              for n in names],
    }
    PREDICTIONS_OUT.write_text(
        json.dumps([{"outputs": [{"value": inner}]}], ensure_ascii=False, indent=2))
    print(f"[predict] wrote {len(names)} entries to {PREDICTIONS_OUT}", flush=True)


# ---------- volume cache ----------
def _volume_path(name: str) -> Path:
    return VOLUME_DIR / f"{name}.npz"


def _load_or_preprocess_volume(name: str, nii_path: Path):
    from pipeline.rave_preprocess import preprocess_nii

    path = _volume_path(name)
    if path.exists():
        with np.load(path) as z:
            return z["volume"]
    with _staged_input(nii_path) as local:
        return preprocess_nii(local, name)


# ---------- input staging ----------
# Submission 3190537e ran stage A at ~40 s/vol on gpu-a100-40 while the identical
# code path measures 0.78 s/vol locally and 1.6 s/vol inside this image on a
# local A100. CPU (down to 2 cores) and RAM (down to 12 GB) were both measured
# and neither reproduces it, which leaves the mounts. If /input is a network or
# FUSE mount, the gzip reader's many small reads are the worst possible access
# pattern for it, while one bulk copy is the best. Copying costs ~1 s when
# /input is fast and saves tens of seconds when it is not, and the bytes handed
# to the decoder are identical either way, so this cannot change a single output.
@contextlib.contextmanager
def _staged_input(nii_path: Path):
    if not INPUT_STAGE:
        yield nii_path
        return
    local = STAGE_DIR / nii_path.name
    try:
        STAGE_DIR.mkdir(parents=True, exist_ok=True)
        t = time.time()
        shutil.copyfile(nii_path, local)
        dt, mb = time.time() - t, local.stat().st_size / 2**20
        _prof.note("stage_in", dt)
        if _prof.n < 4:
            print(f"[stage A]   staged {mb:.0f} MB from {INPUT_DIR} in {dt:.2f}s "
                  f"({mb/max(dt,1e-6):.0f} MB/s)", flush=True)
    except OSError as e:
        # /tmp full or unwritable. The staged copy is a pure I/O optimisation --
        # the decoder sees the same bytes either way -- so read in place rather
        # than kill a 20-hour run over it. Loud, because it costs speed.
        print(f"[stage A] WARNING: input staging to {STAGE_DIR} failed ({e}); "
              f"reading from {INPUT_DIR} directly", file=sys.stderr, flush=True)
        local.unlink(missing_ok=True)
        yield nii_path
        return
    try:
        yield local
    finally:
        local.unlink(missing_ok=True)


class _Prof:
    """Per-step stopwatch for stage A. The platform gives us ~1,995 log lines
    total, so print one compact line per volume and a running median."""

    def __init__(self):
        self.rows, self.n, self.extra, self._cur = [], 0, {}, None

    def note(self, name, dt):
        # Called from inside a lap (input staging happens under "rave"), so
        # fold it into the current row; self.extra is only the pre-lap fallback.
        tgt = self._cur if getattr(self, "_cur", None) is not None else self.extra
        tgt[name] = tgt.get(name, 0.0) + dt

    def start(self):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        self._t0 = self._t = time.time()
        self._cur = dict(self.extra)
        self.extra = {}

    def mark(self, name):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        now = time.time()
        self._cur[name] = self._cur.get(name, 0.0) + (now - self._t)
        self._t = now

    def end(self):
        cur = self._cur
        self.rows.append(cur)
        self.n += 1
        self._cur = None
        return cur, time.time() - self._t0

    def fmt(self, cur):
        return " ".join(f"{k}={v:.2f}" for k, v in cur.items() if v >= 0.005)


_prof = _Prof()


def print_host_report() -> None:
    """One block, printed once. Whatever is making stage A 25x slower on the
    platform than on identical local hardware is environmental, and these are
    the environment facts the container can see for itself."""
    import subprocess
    print("[host] ---- execution environment ----", flush=True)
    try:
        print(f"[host] cpus={os.cpu_count()} affinity={len(os.sched_getaffinity(0))} "
              f"torch_threads={torch.get_num_threads()}", flush=True)
        mem = dict(l.split(":") for l in
                   Path("/proc/meminfo").read_text().splitlines()[:3])
        print(f"[host] mem {' '.join(k+'='+v.strip() for k, v in mem.items())}",
              flush=True)
    except Exception as e:                      # diagnostics must never fail a run
        print(f"[host] cpu/mem probe failed: {e}", flush=True)
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"[host] gpu={p.name} sm={p.major}.{p.minor} mp={p.multi_processor_count} "
              f"vram={total/2**30:.1f}GiB free={free/2**30:.1f}GiB", flush=True)
    else:
        print("[host] gpu=NONE — running on CPU", flush=True)
    try:
        out = subprocess.run(["df", "-hT", str(INPUT_DIR), str(CHECKPOINT_DIR),
                              str(WEIGHTS_DIR), "/tmp"],
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.strip().splitlines():
            print(f"[host] df {line}", flush=True)
    except Exception as e:
        print(f"[host] df failed: {e}", flush=True)
    print("[host] ---------------------------------", flush=True)


# ---------- stage A ----------
def run_stage_a(inputs: list[Path], names: list[str], state: dict) -> bool:
    """Templates + visual cache for every volume. False if SIGTERM interrupted."""
    from pipeline import cls_head, encoder, ret_head, visual_cache
    from pipeline.rave_preprocess import volume_to_model_tensor

    done = set(state["stage_a_done"])
    todo = [p for p in inputs if derive_image_name(p) not in done]

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    if not todo:
        print("[stage A] nothing to do", flush=True)
        return True

    print(f"[stage A] {len(todo)} volume(s) to process", flush=True)
    t0 = time.time()
    enc = encoder.load_encoder(WEIGHTS_DIR, device=DEVICE)
    probe = cls_head.load_probe_head(WEIGHTS_DIR / cls_head.PROBE_HEAD_NAME)
    thresholds = cls_head.load_thresholds(WEIGHTS_DIR / cls_head.THRESHOLDS_NAME)
    sentences = cls_head.load_sentences()
    rhead = ret_head.load_ret_head(WEIGHTS_DIR / "ret_head.pt", device=DEVICE)
    gallery_embeds, gallery_texts = ret_head.load_gallery(WEIGHTS_DIR, device=DEVICE)
    overrides = ret_head.load_display_overrides()
    vision = visual_cache.load_vision_path(
        WEIGHTS_DIR, num_visual_tokens=NUM_VISUAL_TOKENS, device=DEVICE)
    print(f"[stage A] models loaded in {time.time()-t0:.1f}s", flush=True)

    for i, nii_path in enumerate(todo, 1):
        if _stop_requested:
            save_state(state)
            return False
        name = derive_image_name(nii_path)
        _prof.start()

        volume = _load_or_preprocess_volume(name, nii_path)
        _prof.mark("rave")
        vol_tensor = volume_to_model_tensor(volume)
        _prof.mark("tensor")

        feats = encoder.extract_features(enc, vol_tensor, device=DEVICE)
        _prof.mark("tower1")
        cls_entry = cls_head.build_cls_entry(probe, sentences, thresholds,
                                             feats["pooled"], name)
        ret_entry = ret_head.build_ret_entry(
            rhead, gallery_embeds, gallery_texts, feats["activ"], name,
            overrides, device=DEVICE, prob_threshold=RET_THRESHOLD)
        del feats
        _prof.mark("clsret")

        emb = visual_cache.encode_visual(vision, vol_tensor, device=DEVICE)
        _prof.mark("tower2")

        # One 1 MB file per volume, NOT a row of a shared (N, 128, 4096) memmap.
        # The memmap version called .flush() after every volume, which is msync
        # over the whole 2 GB mapping. On a POSIX filesystem that only writes
        # the dirty page (measured: 5 ms local, 11 ms NFS) but an
        # object-storage FUSE mount has no partial writeback -- it re-uploads
        # the entire object. 2 GB per volume at ~65 MB/s is ~31 s, which is the
        # floor submission 3190537e actually showed (31-59 s per volume, flat
        # from volume 1, on code that measures 0.78 s/vol locally). Stage B
        # assembles these into the contiguous cache it wants, once.
        np.save(VISUAL_DIR / f"{name}.npy", emb.numpy())
        del emb

        state["cls_entries"][name] = cls_entry
        state["ret_entries"][name] = ret_entry
        state["stage_a_done"].append(name)
        if i % CKPT_EVERY == 0 or i == len(todo):
            save_state(state)
        _prof.mark("ckpt")

        _volume_path(name).unlink(missing_ok=True)
        torch.cuda.empty_cache()
        cur, t_vol = _prof.end()
        print(f"[stage A] [{i:>4}/{len(todo)}] {name} ({t_vol:.1f}s "
              f"[{_prof.fmt(cur)}], "
              f"cls {sum(1 for k, v in cls_entry.items() if k.startswith('status_') and v == 'positive')} pos, "
              f"ret {len(ret_entry['pred_key_phrases'])} phrases)", flush=True)

    save_state(state)
    del enc, probe, rhead, gallery_embeds, vision
    torch.cuda.empty_cache()
    return True


def _assemble_visual(names: list[str]) -> np.ndarray:
    """Fold stage A's per-volume .npy files into the (N, 128, 4096) cache that
    stage B indexes by row. Written once, sequentially, instead of 2,000 times.

    Idempotent: a complete cache of the right shape is reused as-is, so a resume
    after stage B started does not rebuild it.
    """
    shape = (len(names), NUM_VISUAL_TOKENS, HIDDEN)
    if VISUAL_PATH.exists():
        arr = np.lib.format.open_memmap(VISUAL_PATH, mode="r")
        if arr.shape == shape:
            return arr
        raise RuntimeError(
            f"{VISUAL_PATH} has shape {arr.shape} but this run has {len(names)} "
            f"volumes and {NUM_VISUAL_TOKENS} visual tokens; the checkpoint "
            f"belongs to a different input set")

    t0 = time.time()
    out = np.lib.format.open_memmap(VISUAL_PATH, mode="w+", dtype=np.float16,
                                    shape=shape)
    for row, name in enumerate(names):
        f = VISUAL_DIR / f"{name}.npy"
        if not f.exists():
            raise RuntimeError(
                f"stage A left no visual embedding for {name} ({f}); refusing to "
                f"generate from a zero-filled cache row")
        out[row] = np.load(f)
    out.flush()
    print(f"[stage B] assembled visual cache {shape} in {time.time()-t0:.1f}s",
          flush=True)
    return out


# ---------- stage B ----------
def _adapter_path(slice_name: str) -> Path:
    """A draw's 144 `lora_*` tensors, 15 MB, from the image or from weights.zip.

    Adapters added after the weights were last uploaded ship inside the image
    instead, so that `--reuse-weights <prior submission>` keeps working and the
    37.9 GB weights blob is not re-uploaded for 15 MB of new tensors.
    """
    for cand in (ROOT / "assets" / "lora" / f"{slice_name}.safetensors",
                 WEIGHTS_DIR / "lora" / f"{slice_name}.safetensors"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no adapter {slice_name}.safetensors in {ROOT/'assets'/'lora'} or "
        f"{WEIGHTS_DIR/'lora'}")


def _draw_path(idx: int) -> Path:
    return DRAW_DIR / f"{idx:02d}.json"


def run_stage_b(names: list[str], state: dict) -> bool:
    """Ten draws over one resident model. False if SIGTERM interrupted."""
    from pipeline import ctrg3d_infer

    DRAW_DIR.mkdir(parents=True, exist_ok=True)
    pending = [i for i in range(N_CTRG_DRAWS)
               if state["draws_done"].get(str(i)) != len(names)]
    if not pending:
        print("[stage B] all CTRG-3D draws already complete", flush=True)
        return run_textonly_draw(names, state) if TEXTONLY_DRAW else True

    cache = _assemble_visual(names)
    t0 = time.time()
    lm, builder, cached = ctrg3d_infer.load_ctrg3d(
        WEIGHTS_DIR, cache, num_visual_tokens=NUM_VISUAL_TOKENS,
        max_seq=MAX_SEQ, device=DEVICE)
    tok = builder.tokenizer
    print(f"[stage B] CTRG-3D loaded in {time.time()-t0:.1f}s", flush=True)

    # Prompts are draw-independent: build once.
    prompts, order = {}, []
    for row, n in enumerate(names):
        prompts[n] = builder(
            ctrg3d_infer.render_cls_template(state["cls_entries"].get(n)),
            ctrg3d_infer.render_ret_template(state["ret_entries"].get(n)),
            row)
    lengths = {n: len(tok(prompts[n], add_special_tokens=False)["input_ids"])
               for n in names}
    # Longest-first, as upstream: a batch's slowest row should not stall a batch
    # of short ones. This also fixes the order in which the sampler consumes RNG.
    order = sorted(names, key=lambda n: -lengths[n])
    row_of = {n: i for i, n in enumerate(names)}

    for idx in range(N_CTRG_DRAWS):
        tag, slice_name, overrides = DRAWS[idx]
        path = _draw_path(idx)
        got = json.loads(path.read_text()) if path.exists() else {}
        if len(got) == len(names):
            print(f"[draw {idx}] {tag}: already complete", flush=True)
            continue

        adapter = _adapter_path(slice_name)
        loaded = ctrg3d_infer.load_lora_adapter(lm, adapter)
        sampler = dict(DEFAULT_SAMPLER, **overrides)
        # Upstream ran one process per draw, each seeding 42 at start.
        torch.manual_seed(SEED)
        print(f"[draw {idx}] {tag}  adapter={slice_name} ({loaded} LoRA tensors)  "
              f"sampler={sampler}  resuming at {len(got)}/{len(names)}", flush=True)

        todo = [n for n in order if n not in got]
        t_d = time.time()
        for b in range(0, len(todo), GEN_BATCH):
            if _stop_requested:
                path.write_text(json.dumps(got, ensure_ascii=False))
                state["draws_done"][str(idx)] = len(got)
                save_state(state)
                return False
            chunk = todo[b:b + GEN_BATCH]
            texts = ctrg3d_infer.generate_batch(
                lm, cached, tok, [prompts[n] for n in chunk],
                [row_of[n] for n in chunk], sampler, device=DEVICE)
            for n, t in zip(chunk, texts):
                got[n] = ctrg3d_infer.postprocess_report(t)
            path.write_text(json.dumps(got, ensure_ascii=False))
            state["draws_done"][str(idx)] = len(got)
            save_state(state)
            el = time.time() - t_d
            n_done = len(got) - (len(names) - len(todo))
            print(f"[draw {idx}] {len(got)}/{len(names)}  "
                  f"{el/max(n_done,1):.2f} s/vol", flush=True)

        empty = sum(1 for n in names if not got[n].strip())
        if empty:
            raise RuntimeError(
                f"draw {idx} ({tag}) produced {empty} empty reports; a blank draw "
                f"would silently drag the vote down")

    if cached.n_injected == 0:
        raise RuntimeError("no visual features were injected — the run was blind")
    print(f"[stage B] visual injections {cached.n_injected}", flush=True)

    del lm, cached, cache
    torch.cuda.empty_cache()

    return run_textonly_draw(names, state) if TEXTONLY_DRAW else True


def run_textonly_draw(names: list[str], state: dict) -> bool:
    """Draw index N_CTRG_DRAWS: stock Qwen3-8B + text-only LoRA, no vision.

    Loaded only after the CTRG-3D model is freed, so the two 8B models never
    sit on the card together. Prompts reuse stage A's cls/ret blocks verbatim
    -- the text-only model was trained on exactly those.
    """
    from pipeline import ctrg3d_infer, textonly_infer

    idx = N_CTRG_DRAWS
    path = _draw_path(idx)
    got = json.loads(path.read_text()) if path.exists() else {}
    if len(got) == len(names):
        print(f"[draw {idx}] textonly: already complete", flush=True)
        return True

    adapter_dir = ROOT / "assets" / TEXTONLY_ADAPTER
    t0 = time.time()
    model, tok = textonly_infer.load_textonly(WEIGHTS_DIR, adapter_dir, device=DEVICE)
    print(f"[draw {idx}] textonly loaded in {time.time()-t0:.1f}s", flush=True)

    prompts = {
        n: textonly_infer.build_prompt(
            tok,
            ctrg3d_infer.render_cls_template(state["cls_entries"].get(n)),
            ctrg3d_infer.render_ret_template(state["ret_entries"].get(n)))
        for n in names
    }
    lengths = {n: len(tok(prompts[n], add_special_tokens=False)["input_ids"])
               for n in names}
    order = sorted(names, key=lambda n: -lengths[n])   # longest-first, as upstream

    sampler = dict(textonly_infer.DEFAULT_SAMPLER, max_new_tokens=TEXTONLY_MAX_NEW)
    torch.manual_seed(SEED)
    print(f"[draw {idx}] textonly  adapter={TEXTONLY_ADAPTER}  sampler={sampler}  "
          f"max_seq={TEXTONLY_MAX_SEQ}  resuming at {len(got)}/{len(names)}", flush=True)

    todo = [n for n in order if n not in got]
    t_d = time.time()
    for b in range(0, len(todo), TEXTONLY_BATCH):
        if _stop_requested:
            path.write_text(json.dumps(got, ensure_ascii=False))
            state["draws_done"][str(idx)] = len(got)
            save_state(state)
            return False
        chunk = todo[b:b + TEXTONLY_BATCH]
        texts = textonly_infer.generate_batch(
            model, tok, [prompts[n] for n in chunk], sampler,
            max_seq=TEXTONLY_MAX_SEQ, device=DEVICE)
        for n, t in zip(chunk, texts):
            got[n] = ctrg3d_infer.postprocess_report(t)
        path.write_text(json.dumps(got, ensure_ascii=False))
        state["draws_done"][str(idx)] = len(got)
        save_state(state)
        el = time.time() - t_d
        n_done = len(got) - (len(names) - len(todo))
        print(f"[draw {idx}] {len(got)}/{len(names)}  "
              f"{el/max(n_done,1):.2f} s/vol", flush=True)

    empty = sum(1 for n in names if not got[n].strip())
    if empty:
        raise RuntimeError(
            f"textonly draw produced {empty} empty reports; a blank draw would "
            f"silently drag the vote down")

    del model
    torch.cuda.empty_cache()
    return True


# ---------- stages C/D/E ----------
def run_ensemble(names: list[str]) -> dict:
    from pipeline import ensemble
    from pipeline.radbert import RadBertLabeller

    draws = [json.loads(_draw_path(i).read_text()) for i in range(N_DRAWS)]
    for i, d in enumerate(draws):
        missing = [n for n in names if n not in d]
        if missing:
            raise RuntimeError(f"draw {i} is missing {len(missing)} exams, "
                               f"e.g. {missing[:3]}")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    doc_path = LABEL_DIR / "doc_labels.npy"
    sent_path = LABEL_DIR / "sent_labels.npz"

    t0 = time.time()
    labeller = RadBertLabeller(WEIGHTS_DIR, device=DEVICE)
    print(f"[stage C] RadBERT loaded in {time.time()-t0:.1f}s", flush=True)

    # --- document labels: the vote's input ---
    if doc_path.exists():
        doc_labels = np.load(doc_path)
        print(f"[stage C] reusing {doc_path}", flush=True)
    else:
        doc_labels = np.stack([
            labeller.label_documents([d[n] for n in names]) for d in draws])
        np.save(doc_path, doc_labels)
    print(f"[stage C] doc labels {doc_labels.shape}  "
          f"mean positives/exam/draw {doc_labels.sum(-1).mean():.2f}", flush=True)

    # --- candidate sentence pool ---
    t0 = time.time()
    per, flat = ensemble.build_pool(names, draws)
    print(f"[stage D] pool {len(flat)} sentences ({len(flat)/len(names):.1f}/exam) "
          f"in {time.time()-t0:.1f}s", flush=True)

    if sent_path.exists():
        z = np.load(sent_path, allow_pickle=True)
        if list(z["sents"]) == flat:
            sent_prob = z["prob"]
            print(f"[stage C] reusing {sent_path}", flush=True)
        else:
            sent_prob = None
    else:
        sent_prob = None
    if sent_prob is None:
        sent_prob = labeller.label_sentences(flat)
        np.savez_compressed(sent_path, prob=sent_prob,
                            sents=np.array(flat, dtype=object))
    sent_lab = sent_prob >= 0.5
    print(f"[stage C] label-free sentences {(~sent_lab.any(1)).mean()*100:.1f}%",
          flush=True)

    merged = ensemble.merge(
        names, draws, doc_labels, sent_lab, per,
        min_support=MIN_SUPPORT, backbone=BACKBONE_IDX,
        max_add=MERGE_MAX_ADD, max_add_impression=MERGE_MAX_ADD_IMP)
    (WORK_DIR / "merged.json").write_text(json.dumps(merged, ensure_ascii=False))
    print(f"[stage D] merged {len(merged)} exams  mean {np.mean([len(v) for v in merged.values()]):.0f} chars",
          flush=True)

    final = ensemble.verify_loop(
        names, merged, doc_labels, sent_lab, per, labeller.label_documents,
        min_support=MIN_SUPPORT, iters=VLOOP_ITERS, max_add=VLOOP_MAX_ADD)
    print(f"[stage E] final mean {np.mean([len(v) for v in final.values()]):.0f} chars",
          flush=True)
    return final


# ---------- main ----------
def main() -> int:
    if QUIET_RVE:
        # ~10 INFO lines per volume; the platform keeps only the first
        # ~1,995 log lines, so this is what hid everything after volume 207.
        import logging
        logging.getLogger("vision_engine").setLevel(logging.WARNING)
    print_host_report()
    for d in (OUTPUT_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[predict] FATAL: input dir not found: {INPUT_DIR}", file=sys.stderr)
        return 1
    inputs = sorted(p for p in INPUT_DIR.rglob("*.nii.gz") if p.is_file())
    if not inputs:
        print(f"[predict] no *.nii.gz under {INPUT_DIR}", file=sys.stderr)
        return 1
    names = [derive_image_name(p) for p in inputs]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate input_image_name derived from /input")

    if not 1 <= BACKBONE_IDX < N_DRAWS:
        raise RuntimeError(
            f"N_DRAWS={N_DRAWS} excludes the skeleton draw at index "
            f"{BACKBONE_IDX}; the merge has no prose source")

    state = load_state()
    print(f"[predict] {len(inputs)} volume(s); stage A done for "
          f"{len(state['stage_a_done'])}; draws done "
          f"{ {k: v for k, v in state['draws_done'].items()} }", flush=True)

    if not run_stage_a(inputs, names, state):
        print("[predict] interrupted during stage A; state saved", flush=True)
        return 0
    if not run_stage_b(names, state):
        print("[predict] interrupted during stage B; state saved", flush=True)
        return 0

    final = run_ensemble(names)
    write_final_predictions(final, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
