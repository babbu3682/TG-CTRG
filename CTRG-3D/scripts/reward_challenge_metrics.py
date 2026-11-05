"""Per-rollout reward for the GRPO arm: macro-F1 marginal contribution + BLEU floor.

    # verify before spending GPU-days on it (CPU only, no model weights needed
    # for the F1 half; the RadBERT half needs the checkpoint):
    uv run --no-project --python .venv/bin/python python scripts/reward_challenge_metrics.py --selftest

Design is fixed in docs/2026-08-21-grpo-crg-design.md (§2.4). Two terms, one per
INDEPENDENT measurement channel:

    R_i = ΔF1macro_i / σ_F1  −  λ_B · max(0, BLEU_ref[name] − BLEU_i) / σ_B

WHY macro-F1 AND NOT CRG
------------------------
Both are read off the same RadBERT 18-label vector -- measured correlation of
their per-exam marginal contributions is r = 0.747 -- so they are not
independent evidence, and only one belongs in the reward. Which one matters:

    intervention (assert these labels unconditionally)   ΔCRG      Δmacro-F1
    common top-3                                        +0.0026    -0.0239
    common top-5                                        +0.0082    -0.0223
    rare  bottom-3                                      -0.0488    -0.0213
    rare  bottom-5                                      -0.0734    -0.0463

CRG has a free direction (assert whatever is prevalent: r = 2.1275 makes any
label with p > 1/(2r+1) = 19.03% +EV, and 9 of 18 qualify). macro-F1 has none --
it weights labels equally, so blindly asserting a rare label destroys the
precision that was already its binding constraint. Optimising macro-F1 pulls CRG
along anyway (r = 0.747; the 15-epoch extension moved CRG 0.4459 -> 0.4551 and
macro-F1 0.4300 -> 0.4528 together), so CRG stays an evaluation metric.

And BLEU cannot substitute for the missing guard: appending a report-style
sentence that asserts the 5 most prevalent findings RAISES the official corpus
BLEU_mean by +0.0193. `CRG + BLEU` is the one pair where nothing in the reward
opposes prevalence gaming.

WHY macro-F1 NEEDS A MARGINAL CONTRIBUTION AND CRG WOULD NOT
------------------------------------------------------------
GRPO scores one rollout at a time, but macro-F1 does not decompose per exam: for
a single report TP_j is 0 or 1, and a label that is absent from both prediction
and ground truth leaves TP=FP=FN=0, so F1_j is undefined. The quantity that IS
per-rollout is the derivative against running corpus counts:

    reward_F1 = macroF1(counts + this rollout) − macroF1(counts)

That is not a surrogate; it is the challenge metric's exact marginal value. It
also weights rare findings up automatically -- one recovered Pericardial
effusion is worth 11.4x one recovered Lung nodule (+6.22e-4 vs +0.54e-4).

TWO THINGS THAT SILENTLY BREAK THIS
-----------------------------------
1. Updating the running counts per rollout. The 8 rollouts of one group would
   then see different corpus counts, so their rewards would depend on iteration
   order and the group-relative advantage would be polluted. `commit()` is
   therefore separate from `__call__` and belongs at the step boundary.
2. Recomputing the normalisers from the batch. σ_F1 and σ_B are corpus
   constants; making them batch statistics turns the reward non-stationary and
   rollouts from different steps stop being comparable.

Both are asserted or structurally prevented below rather than left to care.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# Must match reportgen_evaluation/src/crg_score.py's LABEL_COLS, in order.
LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]
N_LABELS = len(LABEL_COLS)

# ---------------------------------------------------------------------------
# Normalisers. Measured, not guessed -- see PROVENANCE below and --selftest,
# which recomputes them from the reference files and fails if they have drifted.
#
# PROVENANCE (2026-08-21, results/ep15best/ctrg3d-qwen3-8B-pillar0os-lz4-vt128):
#   SIGMA_F1: stdev over the split's 1,552 exams of
#             macroF1(corpus counts) − macroF1(counts without that exam)
#   SIGMA_B : stdev of per-sample BLEU_mean from pycocoevalcap Bleu(4)
#             (that split's per-sample mean is 0.2966; corpus BLEU_mean 0.3214,
#              against the official scorer's 0.3194 -- a tokenisation gap)
#
# GRPO divides advantages by the within-group standard deviation, so the
# absolute scale of the reward is irrelevant; only the RATIO between the two
# terms is. Unit-variance per term is what makes "one vote per metric" true.
# ---------------------------------------------------------------------------
SIGMA_F1 = 1.8758e-4
SIGMA_B = 0.1726
LAMBDA_BLEU = 2.0

SELFTEST_TOL = 0.02  # 2% drift in a normaliser is a real change, not noise

DEFAULT_ORG_LABELS = {
    "train": "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/train_predicted_labels.csv",
    "valid": "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/valid_predicted_labels.csv",
}
DEFAULT_REF_PREDICTIONS = (
    _REPO / "results/ep15best/ctrg3d-qwen3-8B-pillar0os-lz4-vt128/"
            "vlm3d_eval_full/predictions.json"
)
DEFAULT_RADBERT_CKPT = "/data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth"
DEFAULT_RADBERT_DIR = (
    "/data/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/"
    "reportgen_evaluation/assets/radbert_local"
)


# ---------------------------------------------------------------------------
# macro-F1 and its marginal contribution
# ---------------------------------------------------------------------------

def macro_f1(tp: Sequence[int], fp: Sequence[int], fn: Sequence[int]) -> float:
    """The challenge's macro-F1: per-label F1 over the corpus, then averaged.

    A label with no predictions and no positives contributes 0, which is what
    the scorer's sklearn call does with zero_division=0.
    """
    total = 0.0
    for j in range(N_LABELS):
        p_den, r_den = tp[j] + fp[j], tp[j] + fn[j]
        p = tp[j] / p_den if p_den else 0.0
        r = tp[j] / r_den if r_den else 0.0
        total += 2 * p * r / (p + r) if (p + r) else 0.0
    return total / N_LABELS


def _classify_counts(pred: Sequence[int], gt: Sequence[int]) -> "tuple[list[int], list[int], list[int]]":
    """One exam's per-label (tp, fp, fn) indicator vectors."""
    tp = [0] * N_LABELS
    fp = [0] * N_LABELS
    fn = [0] * N_LABELS
    for j in range(N_LABELS):
        if pred[j] and gt[j]:
            tp[j] = 1
        elif pred[j]:
            fp[j] = 1
        elif gt[j]:
            fn[j] = 1
    return tp, fp, fn


class MacroF1Marginal:
    """macroF1(running counts + one exam) − macroF1(running counts).

    The running counts are the corpus context this rollout is scored against.
    They are advanced only by `commit()`, never by scoring, so every rollout in
    a GRPO group is measured against an identical context.
    """

    def __init__(self, tp: Sequence[int], fp: Sequence[int], fn: Sequence[int]):
        if not (len(tp) == len(fp) == len(fn) == N_LABELS):
            raise ValueError(f"counts must each have {N_LABELS} entries")
        self.tp, self.fp, self.fn = list(tp), list(fp), list(fn)
        self._pending: List["tuple[list[int], list[int], list[int]]"] = []

    @property
    def value(self) -> float:
        return macro_f1(self.tp, self.fp, self.fn)

    def delta(self, pred: Sequence[int], gt: Sequence[int]) -> float:
        base = macro_f1(self.tp, self.fp, self.fn)
        dtp, dfp, dfn = _classify_counts(pred, gt)
        with_it = macro_f1(
            [self.tp[j] + dtp[j] for j in range(N_LABELS)],
            [self.fp[j] + dfp[j] for j in range(N_LABELS)],
            [self.fn[j] + dfn[j] for j in range(N_LABELS)],
        )
        return with_it - base

    def stage(self, pred: Sequence[int], gt: Sequence[int]) -> None:
        """Remember an exam's contribution without letting it change scoring."""
        self._pending.append(_classify_counts(pred, gt))

    def commit(self) -> int:
        """Fold every staged contribution into the running counts. Step boundary only."""
        n = len(self._pending)
        for dtp, dfp, dfn in self._pending:
            for j in range(N_LABELS):
                self.tp[j] += dtp[j]
                self.fp[j] += dfp[j]
                self.fn[j] += dfn[j]
        self._pending.clear()
        return n


# ---------------------------------------------------------------------------
# BLEU -- the official implementation, not a lookalike
# ---------------------------------------------------------------------------

def bleu_mean_per_item(refs: Sequence[str], hyps: Sequence[str]) -> List[float]:
    """``BLEU_mean`` (the 1-4 average) per item, via the scorer's own Bleu(4).

    `nlg_metrics.py` in reportgen_evaluation scores with
    ``pycocoevalcap.bleu.bleu.Bleu(4)`` and reports the mean of BLEU_1..4 as
    ``BLEU_mean``. Using anything else here would optimise a metric adjacent to
    the one being scored.
    """
    from pycocoevalcap.bleu.bleu import Bleu

    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    keys = [str(i) for i in range(len(refs))]
    gts = {k: [refs[i]] for i, k in enumerate(keys)}
    res = {k: [hyps[i] if hyps[i].strip() else "."] for i, k in enumerate(keys)}
    _, per_n = Bleu(4).compute_score(gts, res)
    return [sum(per_n[n][i] for n in range(4)) / 4 for i in range(len(refs))]


# ---------------------------------------------------------------------------
# label loading
# ---------------------------------------------------------------------------

def load_organizer_labels(path: str) -> Dict[str, List[int]]:
    """``{sample_name: [18]}`` from the organiser's ``*_predicted_labels.csv``.

    Verified 2026-08-21 to agree with the eval pipeline's own
    ``ground_truth.csv`` on all 1,552 valid exams x 18 labels (0 disagreeing
    cells), and to cover all 23,869 train samples of our split. The split
    json's own ``volume_label`` field is NOT used -- it is not accurate.
    """
    out: Dict[str, List[int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in LABEL_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing label columns: {missing}")
        id_field = "VolumeName" if "VolumeName" in (reader.fieldnames or []) else "AccessionNo"
        for row in reader:
            name = row[id_field].replace(".nii.gz", "").replace(".npz", "")
            out[name] = [int(float(row[c])) for c in LABEL_COLS]
    if not out:
        raise ValueError(f"{path} produced no rows")
    return out


def load_reference_reports(predictions_json: str) -> Dict[str, str]:
    """``{sample_name: report}`` from a VLM3D predictions.json."""
    with open(predictions_json, encoding="utf-8") as f:
        blob = json.load(f)

    def find(node):
        if isinstance(node, dict):
            if "generated_reports" in node:
                return node["generated_reports"]
            for v in node.values():
                got = find(v)
                if got:
                    return got
        elif isinstance(node, list):
            for v in node:
                got = find(v)
                if got:
                    return got
        return None

    reports = find(blob)
    if not reports:
        raise ValueError(f"no generated_reports found in {predictions_json}")
    return {r["input_image_name"]: r["report"] for r in reports}


# ---------------------------------------------------------------------------
# the reward itself
# ---------------------------------------------------------------------------

class ChallengeReward:
    """TRL ``reward_funcs`` callable. See the module docstring for the design.

    The KL term is NOT here: ``GRPOConfig(beta=0.04)`` adds it to the objective.
    Putting it in the reward as well would double-count it.
    """

    def __init__(
        self,
        gt_labels: Dict[str, List[int]],
        gt_reports: Dict[str, str],
        bleu_ref: Dict[str, float],
        init_counts: "tuple[Sequence[int], Sequence[int], Sequence[int]]",
        sigma_f1: float = SIGMA_F1,
        sigma_bleu: float = SIGMA_B,
        lambda_bleu: float = LAMBDA_BLEU,
    ):
        self.gt_labels = gt_labels
        self.gt_reports = gt_reports
        self.bleu_ref = bleu_ref
        self.f1 = MacroF1Marginal(*init_counts)
        if sigma_f1 <= 0 or sigma_bleu <= 0:
            raise ValueError("normalisers must be positive")
        self.sigma_f1 = float(sigma_f1)
        self.sigma_bleu = float(sigma_bleu)
        self.lambda_bleu = float(lambda_bleu)
        # Logged per step so the 20-step gate can check that the within-group
        # spread is non-zero -- a zero spread means GRPO learns nothing at all,
        # and it is cheaper to find that out in 20 minutes than in 8 hours.
        self.last_terms: List[Dict[str, float]] = []

    # -- TRL entry point ----------------------------------------------------
    def __call__(self, prompts, completions, sample_name=None, **kwargs) -> List[float]:
        if sample_name is None:
            raise ValueError(
                "the dataset must carry a `sample_name` column: the reward needs "
                "it to look up ground-truth labels and the reference BLEU. TRL "
                "forwards extra dataset columns to reward_funcs as kwargs.")
        texts = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
        if len(texts) != len(sample_name):
            raise ValueError(f"{len(texts)} completions vs {len(sample_name)} names")

        preds = self.predict_labels(texts)

        refs = []
        for name in sample_name:
            if name not in self.gt_reports:
                raise KeyError(f"{name} has no ground-truth report")
            if name not in self.bleu_ref:
                raise KeyError(f"{name} has no reference BLEU; rebuild the cache")
            refs.append(self.gt_reports[name])
        bleus = bleu_mean_per_item(refs, texts)

        out: List[float] = []
        self.last_terms = []
        for pred, name, bleu in zip(preds, sample_name, bleus):
            gt = self.gt_labels.get(name)
            if gt is None:
                raise KeyError(f"{name} is absent from the organiser label file")
            d_f1 = self.f1.delta(pred, gt) / self.sigma_f1
            floor = self.bleu_ref[name]
            pen = self.lambda_bleu * max(0.0, floor - bleu) / self.sigma_bleu
            out.append(d_f1 - pen)
            self.last_terms.append({"f1": d_f1, "bleu": bleu, "bleu_pen": pen})
            # Staged, not committed: every rollout in this group must be scored
            # against the same corpus context.
            self.f1.stage(pred, gt)
        return out

    def commit(self) -> int:
        """Advance the corpus context. Call once per optimiser step, never per rollout."""
        return self.f1.commit()

    # -- RadBERT ------------------------------------------------------------
    def predict_labels(self, texts: Sequence[str]) -> List[List[int]]:
        raise NotImplementedError("attach a labeller with `with_radbert()`")

    def with_radbert(self, radbert_dir: str = DEFAULT_RADBERT_DIR,
                     ckpt: str = DEFAULT_RADBERT_CKPT,
                     device: str = "cuda", threshold: float = 0.5,
                     batch_size: int = 32) -> "ChallengeReward":
        """Bind the scorer's own RadBertClassifier as the label extractor.

        Reusing scripts/label_report_sentences.py's loader on purpose: it already
        handles the classifier.py MODEL_DIR override the scorer does. Whole
        reports (not sentences) are in distribution for this checkpoint.
        """
        import torch
        sys.path.insert(0, str(_HERE))
        from label_report_sentences import classify, load_classifier

        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        tok, model = load_classifier(radbert_dir, ckpt, dev)

        def _predict(texts: Sequence[str]) -> List[List[int]]:
            probs = classify(list(texts), tok, model, dev, batch_size)
            return [[int(p >= threshold) for p in row] for row in probs]

        self.predict_labels = _predict  # type: ignore[assignment]
        return self


def build_bleu_reference(gt_reports: Dict[str, str],
                         ref_reports: Dict[str, str]) -> Dict[str, float]:
    """Per-volume BLEU of the STARTING policy -- the floor the penalty defends.

    A paired floor, not the corpus constant. The split's per-sample BLEU_mean
    averages 0.2966 while corpus BLEU_mean is ~0.32, so a constant 0.3194 floor
    would be active on most samples and quietly turn the one-sided constraint
    into "raise BLEU" -- the opposite of the decision to hold BLEU rather than
    chase it. Against the reference policy's own per-volume score the penalty
    starts at exactly 0 and means "do not get worse than where you started".
    """
    names = [n for n in ref_reports if n in gt_reports]
    if not names:
        raise ValueError("reference and ground-truth reports share no sample_name")
    vals = bleu_mean_per_item([gt_reports[n] for n in names],
                              [ref_reports[n] for n in names])
    return dict(zip(names, vals))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _selftest(args) -> None:
    evald = Path(args.eval_dir)
    gt_csv = evald / "ground_truth.csv"
    inf_csv = evald / "inferred_labels.csv"
    metrics = json.loads((evald / "metrics.json").read_text())

    def read(path):
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        return {r["AccessionNo"]: [int(float(r[c])) for c in LABEL_COLS] for r in rows}

    gt, pred = read(gt_csv), read(inf_csv)
    names = [n for n in gt if n in pred]
    print(f"[data] {len(names)} exams from {evald}")

    # ---- 1. macro_f1 reproduces the scorer -------------------------------
    tp = [0] * N_LABELS; fp = [0] * N_LABELS; fn = [0] * N_LABELS
    for n in names:
        dtp, dfp, dfn = _classify_counts(pred[n], gt[n])
        for j in range(N_LABELS):
            tp[j] += dtp[j]; fp[j] += dfp[j]; fn[j] += dfn[j]
    ours = macro_f1(tp, fp, fn)
    theirs = None
    cls = metrics.get("classification", {})
    for key in ("macro_f1", "macroF1", "macro_F1"):
        if key in cls:
            theirs = float(cls[key]); break
    if theirs is None:
        per = cls.get("per_pathology")
        if per:
            theirs = sum(float(p.get("f1", 0.0)) for p in per) / len(per)
    print(f"[1/4 macro-F1]  ours {ours:.6f}   scorer {theirs if theirs is None else f'{theirs:.6f}'}")
    if theirs is not None:
        assert abs(ours - theirs) < 1e-4, "macro_f1 does not reproduce the scorer"

    # ---- 2. marginal contribution sensitivity ----------------------------
    ctx = MacroF1Marginal(tp, fp, fn)
    base = ctx.value
    sens = {}
    for j, lab in enumerate(LABEL_COLS):
        bumped = macro_f1(
            [tp[k] + (1 if k == j else 0) for k in range(N_LABELS)], fp,
            [fn[k] - (1 if k == j else 0) for k in range(N_LABELS)])
        sens[lab] = bumped - base
    hi = max(sens, key=sens.get); lo = min(sens, key=sens.get)
    ratio = sens[hi] / sens[lo]
    print(f"[2/4 sensitivity] best {hi} {sens[hi]:.3e} / worst {lo} {sens[lo]:.3e} = {ratio:.1f}x")
    assert ratio > 5, "rare labels are not being weighted up; check macro_f1"

    # ---- 3. SIGMA_F1 has not drifted -------------------------------------
    deltas = []
    for n in names:
        dtp, dfp, dfn = _classify_counts(pred[n], gt[n])
        without = macro_f1([tp[j] - dtp[j] for j in range(N_LABELS)],
                           [fp[j] - dfp[j] for j in range(N_LABELS)],
                           [fn[j] - dfn[j] for j in range(N_LABELS)])
        deltas.append(base - without)
    sigma = statistics.stdev(deltas)
    drift = abs(sigma - SIGMA_F1) / SIGMA_F1
    print(f"[3/4 SIGMA_F1]  measured {sigma:.4e}   constant {SIGMA_F1:.4e}   drift {drift:.2%}")
    assert drift < SELFTEST_TOL, (
        f"SIGMA_F1 has drifted {drift:.1%}; the reward's term balance is no longer "
        f"what was calibrated. Update the constant deliberately, with provenance.")

    # ---- 4. SIGMA_B has not drifted, and staging does not leak ------------
    gt_reports = load_reference_reports(evald / "ground_truth.json") \
        if (evald / "ground_truth.json").exists() else None
    ref_reports = load_reference_reports(evald / "predictions.json")
    if gt_reports:
        shared = [n for n in ref_reports if n in gt_reports]
        vals = bleu_mean_per_item([gt_reports[n] for n in shared],
                                  [ref_reports[n] for n in shared])
        sb = statistics.stdev(vals)
        drift_b = abs(sb - SIGMA_B) / SIGMA_B
        print(f"[4/4 SIGMA_B]   measured {sb:.4f} (mean {statistics.mean(vals):.4f})   "
              f"constant {SIGMA_B:.4f}   drift {drift_b:.2%}")
        assert drift_b < SELFTEST_TOL, f"SIGMA_B has drifted {drift_b:.1%}"
    else:
        print("[4/4 SIGMA_B]   skipped (no ground_truth.json in eval_dir)")

    # staging must not change what the next rollout sees
    before = ctx.value
    for n in names[:8]:
        ctx.stage(pred[n], gt[n])
    assert ctx.value == before, (
        "stage() changed the scoring context -- rollouts in one group would be "
        "scored against different corpus counts")
    assert ctx.commit() == 8 and ctx.value != before, "commit() did not advance the context"
    print("[extra] stage() is inert until commit(), commit() advances -- OK")

    print("\nOK -- reward maths reproduces the scorer and the normalisers hold.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="verify the maths against a scored eval directory (CPU only)")
    ap.add_argument("--eval_dir", default=str(
        _REPO / "results/ep15best/ctrg3d-qwen3-8B-pillar0os-lz4-vt128/vlm3d_eval_full"))
    args = ap.parse_args()
    if not args.selftest:
        ap.error("nothing to do; pass --selftest (this module is a library)")
    _selftest(args)


if __name__ == "__main__":
    main()
