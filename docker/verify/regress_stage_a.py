"""Regression gate for the stage-A checkpoint rewrite.

The change replaced a shared (N, 128, 4096) memmap that was `.flush()`ed after
every volume with one 1 MB .npy per volume, assembled once at the start of
stage B. That is an I/O change only, so it must reproduce the previous run
EXACTLY -- not "close", byte-identical, because everything downstream (the ten
draws, the label vote, the merge) is conditioned on these tensors and on the
cls/ret templates.

Reference: verify/run100 (the 100-exam end-to-end run whose scores were the
pre-submission gate) -- its checkpoint/work/visual.npy and checkpoint/state.json
were produced by the previous code on the same 100 input volumes.

Exit code 0 only if visual, cls and ret all match exactly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

DOCKER = Path(__file__).resolve().parent.parent
REF = DOCKER / "verify" / "run100" / "checkpoint"


def main() -> int:
    sys.path.insert(0, str(DOCKER))
    import predict

    inputs = sorted(p for p in predict.INPUT_DIR.rglob("*.nii.gz") if p.is_file())
    names = [predict.derive_image_name(p) for p in inputs]
    print(f"[regress] {len(inputs)} volumes from {predict.INPUT_DIR}", flush=True)

    state = predict.load_state()
    predict.print_host_report()
    if not predict.run_stage_a(inputs, names, state):
        print("[regress] stage A reported interruption", file=sys.stderr)
        return 1
    got_visual = np.asarray(predict._assemble_visual(names))

    ref_visual = np.load(REF / "work" / "visual.npy", mmap_mode="r")
    ref_state = json.loads((REF / "state.json").read_text())

    ok = True

    # --- visual cache: byte-identical or nothing ---
    if got_visual.shape != ref_visual.shape:
        print(f"[regress] FAIL visual shape {got_visual.shape} != "
              f"{ref_visual.shape}", flush=True)
        ok = False
    else:
        ref = np.asarray(ref_visual)
        same = np.array_equal(got_visual.view(np.uint16), ref.view(np.uint16))
        if same:
            print(f"[regress] PASS visual {got_visual.shape} bit-identical "
                  f"({got_visual.nbytes/2**20:.0f} MB)", flush=True)
        else:
            d = np.abs(got_visual.astype(np.float32) - ref.astype(np.float32))
            bad = int((got_visual.view(np.uint16) != ref.view(np.uint16)).sum())
            print(f"[regress] FAIL visual differs: {bad}/{got_visual.size} "
                  f"elements, max|d| {d.max():.3e}", flush=True)
            ok = False

    # --- cls / ret templates: exact JSON equality per exam ---
    for key in ("cls_entries", "ret_entries"):
        miss = [n for n in names if n not in state[key]]
        if miss:
            print(f"[regress] FAIL {key}: {len(miss)} missing, e.g. {miss[:3]}",
                  flush=True)
            ok = False
            continue
        diff = [n for n in names if state[key][n] != ref_state[key][n]]
        if diff:
            print(f"[regress] FAIL {key}: {len(diff)}/{len(names)} differ, "
                  f"e.g. {diff[:3]}", flush=True)
            n = diff[0]
            print(f"           got {json.dumps(state[key][n])[:300]}", flush=True)
            print(f"           ref {json.dumps(ref_state[key][n])[:300]}", flush=True)
            ok = False
        else:
            print(f"[regress] PASS {key} {len(names)}/{len(names)} identical",
                  flush=True)

    # --- the per-volume files the new path writes ---
    n_files = len(list(predict.VISUAL_DIR.glob("*.npy")))
    print(f"[regress] {n_files} per-volume .npy in {predict.VISUAL_DIR} "
          f"(expected {len(names)})", flush=True)
    if n_files != len(names):
        ok = False

    print(f"[regress] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
