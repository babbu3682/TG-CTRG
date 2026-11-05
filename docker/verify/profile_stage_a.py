"""Per-step timing of stage A, to explain the platform's ~40 s/vol.

The submitted container measured 40 s per volume on gpu-a100-40 while the local
100-exam end-to-end run implies ~2 s. This script runs the SAME stage-A code
path as ``predict.py`` on real ``.nii.gz`` input and times every sub-step with
``torch.cuda.synchronize()`` around the GPU work, so the 38 s is attributable.

Run it twice from the sbatch wrapper — once with all cores, once pinned to 4 —
to test whether the gap is CPU starvation inside the platform container.

    PYTHONPATH=<container src...> python verify/profile_stage_a.py N [tag]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DOCKER = Path(__file__).resolve().parent.parent
MAPPING = ("/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/"
           "ctrate_cache_1.25mm_lz4/valid/mapping_path.csv")
WEIGHTS = Path(os.environ.get("WEIGHTS_DIR", DOCKER / "weights"))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TAG = sys.argv[2] if len(sys.argv) > 2 else "default"
WORK = Path(os.environ.get("PROFILE_WORK", "/lscratch/%s/stage_a_profile"
                           % os.environ.get("SLURM_JOB_ID", "0")))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Lap:
    """Accumulating stopwatch keyed by step name."""

    def __init__(self):
        self.rows = []

    def start(self):
        sync()
        self._t = time.time()
        self._cur = {}

    def mark(self, name):
        sync()
        now = time.time()
        self._cur[name] = now - self._t
        self._t = now

    def end(self):
        self.rows.append(self._cur)

    def report(self, tag):
        keys = list(self.rows[0])
        # volume 1 pays one-off lazy-init costs; report it separately
        print(f"\n=== per-step timing [{tag}] (n={len(self.rows)}) ===",
              flush=True)
        hdr = "  ".join(f"{k:>10s}" for k in keys)
        print(f"{'vol':>4s}  {hdr}  {'TOTAL':>10s}")
        for i, r in enumerate(self.rows, 1):
            line = "  ".join(f"{r[k]:10.3f}" for k in keys)
            print(f"{i:4d}  {line}  {sum(r.values()):10.3f}")
        warm = self.rows[1:] or self.rows
        med = {k: float(np.median([r[k] for r in warm])) for k in keys}
        line = "  ".join(f"{med[k]:10.3f}" for k in keys)
        print(f"{'med':>4s}  {line}  {sum(med.values()):10.3f}")
        return med


def main():
    from pipeline import cls_head, encoder, ret_head, visual_cache
    from pipeline.rave_preprocess import preprocess_nii, volume_to_model_tensor
    from pipeline.windowing import batch_apply_ct_windowing

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] tag={TAG} device={dev} "
          f"torch={torch.__version__} nproc={os.cpu_count()} "
          f"affinity={len(os.sched_getaffinity(0))} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"torch_threads={torch.get_num_threads()}", flush=True)
    if dev == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)} "
              f"mem={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB",
              flush=True)

    import csv
    src = {r["sample_name"]: r["source_path"]
           for r in csv.DictReader(open(MAPPING))}
    names = sorted(src)[:N]
    print(f"[env] {len(names)} volumes, first={names[0]}", flush=True)

    t0 = time.time()
    enc = encoder.load_encoder(WEIGHTS, device=dev)
    probe = cls_head.load_probe_head(WEIGHTS / cls_head.PROBE_HEAD_NAME)
    thresholds = cls_head.load_thresholds(WEIGHTS / cls_head.THRESHOLDS_NAME)
    sentences = cls_head.load_sentences()
    rhead = ret_head.load_ret_head(WEIGHTS / "ret_head.pt", device=dev)
    gal_e, gal_t = ret_head.load_gallery(WEIGHTS, device=dev)
    overrides = ret_head.load_display_overrides()
    vision = visual_cache.load_vision_path(WEIGHTS, num_visual_tokens=128,
                                           device=dev)
    print(f"[env] models loaded in {time.time()-t0:.1f}s", flush=True)

    # Same 2 GB memmap the submitted run allocated: 2,000 rows, not len(names).
    WORK.mkdir(parents=True, exist_ok=True)
    vpath = WORK / "visual.npy"
    visual = np.lib.format.open_memmap(vpath, mode="w+", dtype=np.float16,
                                       shape=(2000, 128, 4096))
    state = {"stage_a_done": [], "cls_entries": {}, "ret_entries": {},
             "draws_done": {}, "stage_c_done": False}
    state_path = WORK / "state.json"

    lap = Lap()
    for i, name in enumerate(names):
        lap.start()

        volume = preprocess_nii(src[name], name)
        lap.mark("rave")

        vol_t = volume_to_model_tensor(volume)
        lap.mark("to_tensor")

        # --- tower 1 (pillar-finetune port, fp32 params under autocast) ---
        x = vol_t.unsqueeze(0).to(dev)
        x = batch_apply_ct_windowing(x, "all")
        lap.mark("win1")
        with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=dev.startswith("cuda"), dtype=torch.bfloat16):
            out = enc(x, batch={"anatomy": ["chest_ct"]})
        pooled = out["pooled"].detach().float().cpu()
        activ = out["activ"].detach().float().cpu()
        del out, x
        lap.mark("tower1")

        cls_entry = cls_head.build_cls_entry(probe, sentences, thresholds,
                                             pooled, name)
        lap.mark("cls")

        ret_entry = ret_head.build_ret_entry(rhead, gal_e, gal_t, activ, name,
                                             overrides, device=dev,
                                             prob_threshold=0.4)
        lap.mark("ret")

        # --- tower 2 (PETRG port, bf16) + perceiver + fc ---
        emb = visual_cache.encode_visual(vision, vol_t, device=dev)
        lap.mark("tower2")

        visual[i] = emb.numpy()
        visual.flush()
        lap.mark("memmap")

        state["cls_entries"][name] = cls_entry
        state["ret_entries"][name] = ret_entry
        state["stage_a_done"].append(name)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False))
        os.replace(tmp, state_path)
        lap.mark("save_state")

        torch.cuda.empty_cache()
        lap.mark("empty_cache")

        lap.end()
        print(f"  vol {i+1}/{len(names)} {name} done", flush=True)

    med = lap.report(TAG)
    print(f"\n[{TAG}] median total {sum(med.values()):.2f}s/vol  "
          f"-> 2,000 volumes = {sum(med.values())*2000/3600:.1f} h", flush=True)
    print(f"[{TAG}] platform observed: 40 s/vol -> 22.2 h", flush=True)

    del visual
    vpath.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
