#!/usr/bin/env python3
"""Turn the starting policy's generations on the frozen subset into the three
reference artifacts the crgf1bleu reward needs.

    python scripts/build_grpo_reference_crgf1bleu.py \
        --pred_csv data/grpo_ref_crgf1bleu/ref_train8000.csv

Produces data/grpo_ref_crgf1bleu/reference.json holding:

  bleu_floor[name]   per-volume BLEU_mean of the STARTING policy.
                     The one-sided penalty is measured against this, so it is
                     exactly 0 at the start and only fires when a rollout is
                     WORSE than where the policy began. The 500-step pilot used
                     a single corpus constant (0.2966) instead, which put the
                     starting policy itself in penalty on 65.5% of samples and
                     silently turned "hold BLEU" into "raise BLEU".

  corpus{tp,fp,fn}   per-label counts of the starting policy's OWN predictions
                     on the subset. This is the operating point the macro-F1
                     marginal is differentiated at. The pilot seeded the corpus
                     with the ground truth treated as a perfect prediction, so
                     macro-F1 was exactly 1.0 and the marginal was taken at a
                     singularity: measured std 0.082 against an intended 1.052.

  sigma_f1/s/bleu    the three normalisers, RE-MEASURED on this corpus. They are
                     what makes "one vote per term" true, and they are not
                     portable across corpus sizes -- SIGMA_F1 in
                     reward_challenge_metrics.py was measured on 1,552 valid
                     exams and is wrong for an 8,000-volume train corpus.

  r, U               CRG's GT-only constants for this subset (from the subset
                     meta, re-derived here and cross-checked).

Nothing here is stochastic beyond the generation it reads, so re-running it on
the same CSV reproduces the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))

OUT_DIR = _REPO / "data/grpo_ref_crgf1bleu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_csv", default=str(OUT_DIR / "ref_train8000.csv"))
    ap.add_argument("--subset_json", default=str(OUT_DIR / "subset8000.json"))
    ap.add_argument("--subset_meta", default=str(OUT_DIR / "subset8000_meta.json"))
    ap.add_argument("--organiser_labels", default=(
        "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/train_predicted_labels.csv"))
    ap.add_argument("--out", default=str(OUT_DIR / "reference.json"))
    args = ap.parse_args()

    from reward_challenge_metrics import (LABEL_COLS, N_LABELS, _classify_counts,
                                          bleu_mean_per_item, load_organizer_labels,
                                          macro_f1)
    from label_report_sentences import classify, load_classifier
    import torch

    subset = json.loads(Path(args.subset_json).read_text())
    meta = json.loads(Path(args.subset_meta).read_text())
    gt_labels = load_organizer_labels(args.organiser_labels)

    df = pd.read_csv(args.pred_csv)
    for col in ("SampleName", "GT_report", "Pred_report"):
        if col not in df.columns:
            raise RuntimeError(f"{args.pred_csv} has no '{col}' column")
    by_name = {row["SampleName"]: row for _, row in df.iterrows()}

    # The generation must cover the subset EXACTLY. A hole here becomes a KeyError
    # 40 minutes into training, or worse a silently missing floor.
    absent = [n for n in subset if n not in by_name]
    if absent:
        raise RuntimeError(
            f"{len(absent)} of {len(subset)} subset volumes are missing from "
            f"{args.pred_csv}, e.g. {absent[:3]}. Re-run the reference generation "
            f"with --names_json {args.subset_json}.")

    gts = [str(by_name[n]["GT_report"]) for n in subset]
    preds = [str(by_name[n]["Pred_report"]) for n in subset]

    # ---- BLEU paired floor --------------------------------------------------
    bleu = bleu_mean_per_item(gts, preds)
    bleu_floor = {n: float(b) for n, b in zip(subset, bleu)}
    sigma_bleu = float(np.std(bleu))
    print(f"[bleu] per-volume mean={np.mean(bleu):.4f} median={np.median(bleu):.4f} "
          f"sigma={sigma_bleu:.4f}", flush=True)

    # ---- RadBERT labels of the starting policy's own reports ----------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_classifier(
        "/data/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/"
        "reportgen_evaluation/assets/radbert_local",
        "/data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth", dev)
    probs = classify(preds, tok, model, dev, 32)
    P = (probs >= 0.5).astype(int)
    G = np.array([gt_labels[n] for n in subset])
    print(f"[radbert] labelled {len(P)} reports on {dev}", flush=True)

    tp = [0] * N_LABELS; fp = [0] * N_LABELS; fn = [0] * N_LABELS
    for i in range(len(subset)):
        a, b, c = _classify_counts(P[i], G[i])
        for j in range(N_LABELS):
            tp[j] += a[j]; fp[j] += b[j]; fn[j] += c[j]
    f1_op = macro_f1(tp, fp, fn)

    # ---- CRG constants, re-derived and cross-checked ------------------------
    X = N_LABELS * len(subset)
    A = int(G.sum())
    r = (X - A) / (2 * A)
    U = (X - A) / 2
    if abs(r - meta["r"]) > 1e-9:
        raise RuntimeError(f"r mismatch: subset meta {meta['r']} vs recomputed {r}. "
                           f"The label file or the subset changed under us.")
    TP = int(((P == 1) & (G == 1)).sum()); FN = int(((P == 0) & (G == 1)).sum())
    FP = int(((P == 1) & (G == 0)).sum())
    s_total = r * TP - r * FN - FP
    crg_op = U / (2 * U - s_total)

    # ---- the three normalisers, measured HERE -------------------------------
    s_i = np.array([r * int(((P[i] == 1) & (G[i] == 1)).sum())
                    - r * int(((P[i] == 0) & (G[i] == 1)).sum())
                    - int(((P[i] == 1) & (G[i] == 0)).sum())
                    for i in range(len(subset))])
    sigma_s = float(s_i.std())

    # leave-one-out marginal: exactly what the reward computes at training time,
    # but with the exam already in the corpus, so this is its true spread.
    base = macro_f1(tp, fp, fn)
    d = np.empty(len(subset))
    for i in range(len(subset)):
        a, b, c = _classify_counts(P[i], G[i])
        d[i] = base - macro_f1([tp[j] - a[j] for j in range(N_LABELS)],
                               [fp[j] - b[j] for j in range(N_LABELS)],
                               [fn[j] - c[j] for j in range(N_LABELS)])
    sigma_f1 = float(d.std())

    print(f"[corpus] macro-F1={f1_op:.4f}  CRG={crg_op:.4f}  "
          f"(TP={TP} FN={FN} FP={FP})", flush=True)
    print(f"[sigma]  s={sigma_s:.4f}  f1={sigma_f1:.6e}  bleu={sigma_bleu:.4f}", flush=True)

    if not (sigma_s > 0 and sigma_f1 > 0 and sigma_bleu > 0):
        raise RuntimeError("a normaliser came out non-positive; refusing to write it")

    out = {
        "pred_csv": str(Path(args.pred_csv).resolve()),
        "subset_json": str(Path(args.subset_json).resolve()),
        "n": len(subset),
        "r": r, "U": U, "X": X, "A": A,
        "corpus_tp": tp, "corpus_fp": fp, "corpus_fn": fn,
        "operating_point": {"macro_f1": f1_op, "crg": crg_op,
                            "TP": TP, "FN": FN, "FP": FP},
        "sigma_s": sigma_s, "sigma_f1": sigma_f1, "sigma_bleu": sigma_bleu,
        "bleu_floor": bleu_floor,
        "label_cols": LABEL_COLS,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
