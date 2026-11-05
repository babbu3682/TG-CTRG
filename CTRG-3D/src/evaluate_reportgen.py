#!/usr/bin/env python3
"""
Evaluate a CTRG-3D test.py results CSV against the VLM3D report-generation
challenge's own scorer:
  VLM3D-Dockers/ct_challenges/reportgen_evaluation

This builds the exact input/output artifacts the challenge's eval container
expects (predictions.json / ground_truth.json / ground_truth.csv, matching
reportgen_evaluation/README.md's schema), then calls that package's own
src/nlg_metrics.py to compute BLEU/ROUGE-L/METEOR/CIDEr, rather than
reimplementing the metric math. predictions.json/ground_truth.json are also
directly reusable as-is if the real eval Docker image is run later.

Classification + CRG scoring (src/calc_scores.py, src/crg_score.py) additionally
require a RadBERT label-inference head to infer pathology labels from the
generated report text:
  --radbert_model_path /data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth
  --radbert_dir        VLM3D-Dockers/ct_challenges/reportgen_evaluation/assets/radbert_local
(the classifier weights; the tokenizer+backbone dir was downloaded from
'zzxslp/RadBERT-RoBERTa-4m' into assets/radbert_local -- see reportgen_evaluation/
README.md's asset note). Both flags are optional; omit them to only run NLG.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# VLM3D-Dockers is a sibling of this repo under 0.Challenge/, not a subdirectory
# of it. Overridable with $REPORTGEN_EVAL_DIR.
REPORTGEN_EVAL_DIR = Path(os.environ.get(
    "REPORTGEN_EVAL_DIR",
    "/data/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_evaluation"))
REPORTGEN_SRC = REPORTGEN_EVAL_DIR / "src"
if not REPORTGEN_SRC.is_dir():
    sys.exit(f"reportgen_evaluation not found at {REPORTGEN_EVAL_DIR}; set $REPORTGEN_EVAL_DIR")

# Must match reportgen_evaluation/src/crg_score.py's LABEL_COLS and
# ground_truth.csv's column order exactly (cross-checked against
# pillar-finetune's valid_abn_reports.json "pred_labels" key order).
LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]


def build_predictions_json(df: pd.DataFrame, out_path: Path) -> None:
    generated_reports = [
        {"input_image_name": row["SampleName"], "report": row["Pred_report"]}
        for _, row in df.iterrows()
    ]
    wrapped = [{
        "outputs": [{
            "value": {
                "name": "Generated reports",
                "type": "Report generation",
                "version": {"major": 1, "minor": 0},
                "generated_reports": generated_reports,
            }
        }]
    }]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wrapped, f, ensure_ascii=False, indent=2)


def build_ground_truth_json(df: pd.DataFrame, out_path: Path) -> None:
    generated_reports = [
        {"input_image_name": row["SampleName"], "report": row["GT_report"]}
        for _, row in df.iterrows()
    ]
    payload = {
        "name": "Generated reports",
        "type": "Report generation",
        "generated_reports": generated_reports,
        "version": {"major": 1, "minor": 0},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_ground_truth_csv(sample_names, split_json_path: Path, out_path: Path) -> dict:
    """Rebuilds the challenge's AccessionNo + 18-label CSV from the same
    merged_*.json "volume_label" field used at inference time (the field
    that produced test.sh's json_path).

    Also returns a provenance record describing exactly which labels were used.
    This exists because the label lineage HAS silently changed under us: the
    2026-08-09 04:22 scoring of the joint checkpoints read a version of
    merged_valid_filtered_prob_half.json whose volume_label was
    corrupted (Cardiomegaly positive in 2 of 1,552 exams instead of 161), and
    the file was regenerated with corrected labels at 21:10 the same day. The
    stale ground_truth.csv left behind carried no record of which labels it
    came from, so the mismatch was only found much later by diffing two eval
    directories. Recording the source path, its mtime and a label checksum in
    metrics.json makes that comparison a one-liner next time.
    """
    labels_by_sample = {}
    with open(split_json_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            labels_by_sample[record["sample_name"]] = record["volume_label"]

    rows, missing = [], []
    for name in sample_names:
        if name not in labels_by_sample:
            missing.append(name)
            continue
        rows.append([name] + labels_by_sample[name])
    if missing:
        preview = missing[:5]
        print(f"[warn] {len(missing)} samples missing volume_label in {split_json_path}, "
              f"excluded from ground_truth.csv: {preview}{'...' if len(missing) > 5 else ''}")

    gt_df = pd.DataFrame(rows, columns=["AccessionNo"] + LABEL_COLS)
    gt_df.to_csv(out_path, index=False)

    counts = {c: int(gt_df[c].sum()) for c in LABEL_COLS}
    digest = hashlib.sha256(
        gt_df.sort_values("AccessionNo").to_csv(index=False).encode()
    ).hexdigest()[:16]
    prov = {
        "split_json_path": str(split_json_path),
        "split_json_mtime": datetime.fromtimestamp(
            os.path.getmtime(split_json_path)).isoformat(timespec="seconds"),
        "n_exams": len(gt_df),
        "n_positive": int(sum(counts.values())),
        "positives_per_label": counts,
        "ground_truth_sha256_16": digest,
    }
    print(f"  ground truth: {prov['n_exams']} exams, {prov['n_positive']} positives, "
          f"sha={digest}, from {split_json_path.name} (mtime {prov['split_json_mtime']})")
    return prov


def run_nlg_metrics(pred_json: Path, gt_json: Path, out_json: Path) -> dict:
    sys.path.insert(0, str(REPORTGEN_SRC))
    import nlg_metrics
    nlg_metrics.run(pred_json, gt_json, out_json)
    return json.load(open(out_json))


def run_classification_and_crg(pred_json: Path, gt_csv: Path, radbert_model_path: str,
                                radbert_dir: str, out_dir: Path) -> dict:
    sys.path.insert(0, str(REPORTGEN_SRC))
    # classifier.py AND dataset.py each hardcode their own module-level
    # MODEL_DIR = /opt/app/radbert_local (the Docker image's asset path);
    # override both before RadBertClassifier()/CTDataset() are built so this
    # works from a plain host checkout too.
    import classifier as _classifier_mod
    import dataset as _dataset_mod
    _classifier_mod.MODEL_DIR = Path(radbert_dir).resolve()
    _dataset_mod.MODEL_DIR = Path(radbert_dir).resolve()
    import infer as _infer_mod
    import calc_scores as _calc_scores_mod
    import crg_score as _crg_score_mod

    pred_csv = out_dir / "inferred_labels.csv"
    cls_json = out_dir / "classification_scores.json"
    crg_json = out_dir / "crg_scores.json"

    _infer_mod.run_inference(pred_json, Path(radbert_model_path), pred_csv)
    _calc_scores_mod.evaluate(pred_csv, gt_csv, cls_json)
    _crg_score_mod.main(pred_csv, gt_csv, crg_json)

    return {"classification": json.load(open(cls_json)), "crg": json.load(open(crg_json))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result_csv", required=True, help="test.py's output CSV (SampleName, Question, GT_report, Pred_report).")
    ap.add_argument("--split_json_path", required=True, help="The merged_*.json used as test.sh's json_path (for volume_label -> ground_truth.csv).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--radbert_model_path", default=None, help="Optional: path to RadBertClassifier.pth. Requires --radbert_dir too.")
    ap.add_argument("--radbert_dir", default=None, help="Optional: radbert_local/ tokenizer+backbone dir.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.result_csv)
    df["GT_report"] = df["GT_report"].fillna("")
    df["Pred_report"] = df["Pred_report"].fillna("")
    print(f"Loaded {len(df)} generated reports from {args.result_csv}")

    pred_json = out_dir / "predictions.json"
    gt_json = out_dir / "ground_truth.json"
    gt_csv = out_dir / "ground_truth.csv"
    nlg_json = out_dir / "nlg_scores.json"

    build_predictions_json(df, pred_json)
    build_ground_truth_json(df, gt_json)
    gt_provenance = build_ground_truth_csv(df["SampleName"].tolist(), Path(args.split_json_path), gt_csv)

    print("\n=== NLG metrics (BLEU / ROUGE-L / METEOR / CIDEr) ===")
    nlg_scores = run_nlg_metrics(pred_json, gt_json, nlg_json)
    for k, v in nlg_scores.items():
        print(f"  {k}: {v:.4f}")

    combined = {"generation": nlg_scores, "ground_truth_provenance": gt_provenance}

    if args.radbert_model_path and args.radbert_dir:
        print("\n=== Classification + CRG (RadBERT label inference) ===")
        cls_crg = run_classification_and_crg(pred_json, gt_csv, args.radbert_model_path, args.radbert_dir, out_dir)
        print("  classification (macro):", cls_crg["classification"]["macro"])
        print("  CRG:", cls_crg["crg"]["CRG"])
        combined.update(cls_crg)
    else:
        print("\n[skip] classification + CRG: needs RadBertClassifier.pth + radbert_local/ "
              "(challenge release assets, ~1GB, not in this repo -- see reportgen_evaluation/README.md's "
              "asset note). Pass --radbert_model_path/--radbert_dir once available.")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
