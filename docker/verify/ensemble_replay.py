#!/usr/bin/env python
"""Replay the ranked merge with the CONTAINER's ensemble code and diff it
against `bb1000F5I0_v.csv`, the CSV the submission is named after.

No GPU, no generation: the ten draws already exist on disk from the runs that
produced the ranking, so this isolates stages C/D/E — vote, label-guided merge,
read-back verify loop — and asks whether the port reproduces them.

Two modes, and the difference matters:

  --labels scorer   feed the vote the scorer's own inferred_labels.csv, exactly
                    as label_guided_merge.py did. If the merged text still
                    differs, the bug is in the ported MERGE logic.
  --labels radbert  re-derive the document labels with pipeline/radbert.py, the
                    way the container will have to on a test set that has no
                    scorer. If `scorer` matches and this does not, the bug is in
                    the labeller wiring, not the merge.

Exact text equality is the bar. These are deterministic set operations over
fixed inputs; anything less than 1552/1552 identical means the port drifted.

    sbatch verify/ensemble_replay.sbatch
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DOCKER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOCKER / "src"))

from pipeline import ensemble                                  # noqa: E402

PT = Path("/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D")
RES = PT / "results"

DRAW_TAGS = [
    "grpo_newencB/700", "grpo_newencB/800", "grpo_newencB/800_t10p95",
    "grpo_newencB/800_rep100", "grpo_newencB/800_rep100_t10p95",
    "grpo_newencB/900", "grpo_B_a2from700/1000", "grpo_B_a2from700/900",
    "grpo_B_nobleu/700", "grpo_B_g3/700",
]
BACKBONE = 6
REFERENCE = RES / "grpo_newencB" / "bb1000F5I0_v.csv"
MERGED_REF = RES / "grpo_newencB" / "bb1000F5I0.csv"   # pre-verify-loop stage


def main() -> int:
    global DRAW_TAGS, REFERENCE, MERGED_REF, BACKBONE

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["scorer", "radbert"], default="scorer")
    ap.add_argument("--draws", default="",
                    help="space-separated <sub>/<tag> draw list, in order. "
                         "Order is load-bearing: cluster_block keeps the first "
                         "draw's sentence as each cluster's representative. "
                         "Default = the bb1000F5I0_v set.")
    ap.add_argument("--backbone", type=int, default=BACKBONE)
    ap.add_argument("--out_csv", default="",
                    help="where to write the replayed reports (default verify/replay_<labels>.csv)")
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--max_add", type=int, default=5)
    ap.add_argument("--max_add_imp", type=int, default=-1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--vloop_max_add", type=int, default=2)
    # --- 11-draw (text-only) 검증용. 기본값은 V-2 그대로라 기존 호출 불변 ---
    ap.add_argument("--extra_draw", default="",
                    help="append one more draw tag, e.g. textonly/checkpoint-11928")
    ap.add_argument("--reference", default="",
                    help="override the stage-E reference CSV tag")
    ap.add_argument("--merged_ref", default="",
                    help="override the stage-D reference CSV tag")
    ap.add_argument("--no_vloop", action="store_true",
                    help="stop after stage D (the reference has no verify loop)")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N exams only; the merge is per-exam so a subset "
                         "is still an exact check")
    a = ap.parse_args()

    if a.draws:
        DRAW_TAGS = a.draws.split()
    BACKBONE = a.backbone
    if a.extra_draw:
        DRAW_TAGS = DRAW_TAGS + [a.extra_draw]
    if a.reference:
        REFERENCE = RES / "grpo_newencB" / f"{a.reference}.csv"
    if a.merged_ref:
        MERGED_REF = RES / "grpo_newencB" / f"{a.merged_ref}.csv"
    print(f"[replay] {len(DRAW_TAGS)} draws, backbone {BACKBONE} "
          f"({DRAW_TAGS[BACKBONE]})", flush=True)

    ref = pd.read_csv(REFERENCE).set_index("SampleName")
    names = list(ref.index)
    if a.limit:
        names = names[:a.limit]
    print(f"[replay] reference {REFERENCE.name}: {len(names)} exams", flush=True)

    draws = []
    for tag in DRAW_TAGS:
        df = pd.read_csv(RES / f"{tag}.csv").set_index("SampleName")
        missing = [n for n in names if n not in df.index]
        if missing:
            sys.exit(f"{tag}: missing {len(missing)} exams, e.g. {missing[:3]}")
        draws.append({n: str(df.loc[n, "Pred_report"]) for n in names})

    labeller = None
    if a.labels == "scorer":
        doc = []
        for tag in DRAW_TAGS:
            lp = RES / tag / "vlm3d_eval" / "inferred_labels.csv"
            if not lp.exists():
                sys.exit(f"no inferred_labels for {tag}: {lp}")
            v = pd.read_csv(lp).set_index("AccessionNo")[ensemble.LABEL_COLS]
            doc.append(v.loc[names].values.astype(np.int8))
        doc_labels = np.stack(doc)
    else:
        sys.path.insert(0, str(DOCKER / "src"))
        from pipeline.radbert import RadBertLabeller
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        labeller = RadBertLabeller(DOCKER / "weights", device=dev)
        print(f"[replay] RadBERT on {dev}", flush=True)
        doc_labels = np.stack([labeller.label_documents([d[n] for n in names])
                               for d in draws])
        # Cross-check against the scorer's own labels while we are here.
        ref_doc = np.stack([
            pd.read_csv(RES / tag / "vlm3d_eval" / "inferred_labels.csv")
              .set_index("AccessionNo")[ensemble.LABEL_COLS]
              .loc[names].values.astype(np.int8) for tag in DRAW_TAGS])
        agree = (doc_labels == ref_doc).mean()
        print(f"[replay] doc-label agreement with the scorer: {agree*100:.4f}% "
              f"({int((doc_labels != ref_doc).sum())} cells differ)", flush=True)

    print(f"[replay] doc labels {doc_labels.shape}", flush=True)

    per, flat = ensemble.build_pool(names, draws)
    print(f"[replay] pool {len(flat)} sentences ({len(flat)/len(names):.1f}/exam)",
          flush=True)

    if labeller is None:
        from pipeline.radbert import RadBertLabeller
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        labeller = RadBertLabeller(DOCKER / "weights", device=dev)
        print(f"[replay] RadBERT on {dev}", flush=True)
    sent_lab = labeller.label_sentences(flat) >= 0.5
    print(f"[replay] label-free sentences {(~sent_lab.any(1)).mean()*100:.1f}%",
          flush=True)

    merged = ensemble.merge(names, draws, doc_labels, sent_lab, per,
                            min_support=a.min_support, backbone=BACKBONE,
                            max_add=a.max_add, max_add_impression=a.max_add_imp)

    if MERGED_REF.exists():
        m_ref = pd.read_csv(MERGED_REF).set_index("SampleName")
        same = sum(1 for n in names if merged[n] == str(m_ref.loc[n, "Pred_report"]))
        print(f"[stage D] merge == {MERGED_REF.name}: {same}/{len(names)}", flush=True)

    if a.no_vloop:
        same = sum(1 for n in names if merged[n] == str(ref.loc[n, "Pred_report"]))
        print(f"\n[stage D] final == {REFERENCE.name}: {same}/{len(names)}")
        if same != len(names):
            bad = [n for n in names if merged[n] != str(ref.loc[n, "Pred_report"])][:2]
            for n in bad:
                print(f"\n--- {n} ---\n GOT : {merged[n][:400]}\n WANT: "
                      f"{str(ref.loc[n,'Pred_report'])[:400]}")
        return 0 if same == len(names) else 1

    final = ensemble.verify_loop(names, merged, doc_labels, sent_lab, per,
                                 labeller.label_documents,
                                 min_support=a.min_support, iters=a.iters,
                                 max_add=a.vloop_max_add)

    same = sum(1 for n in names if final[n] == str(ref.loc[n, "Pred_report"]))
    print(f"\n[stage E] final == {REFERENCE.name}: {same}/{len(names)}")
    if same != len(names):
        bad = [n for n in names if final[n] != str(ref.loc[n, "Pred_report"])][:3]
        for n in bad:
            print(f"\n--- {n} ---\n GOT : {final[n][:400]}\n WANT: "
                  f"{str(ref.loc[n, 'Pred_report'])[:400]}")
    out = Path(a.out_csv) if a.out_csv else DOCKER / "verify" / f"replay_{a.labels}.csv"
    pd.DataFrame([{"SampleName": n, "Question": ref.loc[n, "Question"],
                   "GT_report": ref.loc[n, "GT_report"], "Pred_report": final[n]}
                  for n in names]).to_csv(out, index=False)
    print(f"[replay] wrote {out}")

    print("\nVERDICT:", "PASS" if same == len(names) else "FAIL")
    return 0 if same == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
