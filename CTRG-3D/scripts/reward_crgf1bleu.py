"""Per-rollout reward for the crgf1bleu round: CRG + macro-F1 + BLEU floor.

    R_i = alpha * s_i / sigma_S
        + gamma * dF1_i / sigma_F1
        - lambda * max(0, floor_i - BLEU_i) / sigma_B

Design and the measurements behind every constant: docs/2026-08-22-grpo-review.md.
This file does NOT modify scripts/reward_challenge_metrics.py -- that module is
what the 500-step pilot (job 28075386) ran with and it stays reproducible. The
shared metric math is imported from it; only the reward's shape is new here.

WHY CRG IS BACK, AND AS A PER-EXAM TERM
---------------------------------------
The 2026-08-21 design dropped CRG because its per-exam marginal correlates with
macro-F1's at r = 0.747 -- "not independent evidence". That looked at the wrong
axis. Two terms are not there to pool independent evidence; they are there to
cover each other's failure mode, and the 0.25 they do NOT share is exactly the
direction they disagree on, which is prevalence gaming.

And CRG needs no marginal at all. From reportgen_evaluation/src/crg_score.py:

    s   = r*TP - r*FN - FP        <- TP/FN/FP are additive over exams
    CRG = U / (2U - s)            <- monotone in s; r and U are GT-only constants

so `s_i = r*TP_i - r*FN_i - FP_i` IS the per-exam quantity, with no running
corpus and no singularity to differentiate at. Measured spread on the starting
policy: sigma_S ~ 4.2, against a macro-F1 marginal whose spread collapsed to
0.082 in the pilot because the corpus was seeded with the ground truth.

WHY alpha = gamma = 1
---------------------
CRG has a free gaming direction: any label with prevalence above
1/(2r+1) ~ 18.6% is +EV to assert blind, and 9 of 18 qualify. macro-F1 has none
-- it weights labels equally, so asserting a prevalent label blind costs the
precision that was already binding. Measured on the starting policy, asserting
the 5 most prevalent labels unconditionally:

    CRG 0.4551 -> 0.4633 (+0.0082)     macro-F1 0.4528 -> 0.4305 (-0.0224)
    in normalised per-exam units:  d(s)/sigma_S    = +0.0666
                                   d(dF1)/sigma_F1 = -0.1662
    => gaming is a net loss whenever gamma/alpha > 0.401

alpha = gamma = 1 clears that by 2.5x. The ratio is measured, not chosen.

WHY THE CORPUS IS FROZEN
------------------------
The macro-F1 marginal is taken against a corpus that never advances: the
starting policy's own predictions on the frozen subset. `commit()` does not
exist here on purpose. The pilot advanced the corpus every step, so each exam's
marginal shrank as the corpus grew while sigma_F1 stayed fixed -- the term
quietly lost magnitude over the run. A frozen corpus makes the reward stationary,
which is what lets rollouts from step 10 and step 900 be compared at all.

WHY THE BLEU FLOOR IS PAIRED
----------------------------
floor_i is THIS volume's BLEU under the starting policy, so the penalty is
exactly 0 where the run begins and only fires on regression. The pilot used one
corpus constant (0.2966) for every volume; measured, that puts the starting
policy itself in penalty on 65.5% of samples, because per-sample BLEU is
right-skewed (median 0.2508 < mean 0.2966). That silently turned a one-sided
"do not lose BLEU" constraint into "raise BLEU", and BLEU then dominated the
reward 9.4 : 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from reward_challenge_metrics import (LABEL_COLS, N_LABELS, _classify_counts,  # noqa: E402
                                      bleu_mean_per_item, macro_f1)

ALPHA = 1.0
GAMMA = 1.0
LAMBDA_BLEU = 1.0


class CRGF1BLEUReward:
    """TRL ``reward_funcs`` callable. Corpus and normalisers come from a
    reference file built by scripts/build_grpo_reference_crgf1bleu.py.

    The KL term is NOT here; it is GRPOConfig(beta=...)'s job. For this round
    beta is 0 -- see the review doc: the adapter-disable reference is the
    pre-SFT base model, so the KL term pulled the wrong way and at 0.2% of the
    loss it was never a defence anyway.
    """

    def __init__(self, reference_json: str, gt_labels: Dict[str, List[int]],
                 gt_reports: Dict[str, str], alpha: float = ALPHA,
                 gamma: float = GAMMA, lambda_bleu: float = LAMBDA_BLEU):
        ref = json.loads(Path(reference_json).read_text())
        if ref.get("label_cols") != LABEL_COLS:
            raise ValueError(
                f"{reference_json} was built with a different label order; the "
                f"18-vector would be permuted against the scorer's.")
        self.r = float(ref["r"])
        self.tp = list(ref["corpus_tp"])
        self.fp = list(ref["corpus_fp"])
        self.fn = list(ref["corpus_fn"])
        if not (len(self.tp) == len(self.fp) == len(self.fn) == N_LABELS):
            raise ValueError("reference corpus counts are malformed")
        self.base_f1 = macro_f1(self.tp, self.fp, self.fn)
        self.sigma_s = float(ref["sigma_s"])
        self.sigma_f1 = float(ref["sigma_f1"])
        self.sigma_bleu = float(ref["sigma_bleu"])
        if min(self.sigma_s, self.sigma_f1, self.sigma_bleu) <= 0:
            raise ValueError("a normaliser in the reference file is non-positive")
        self.bleu_floor = {k: float(v) for k, v in ref["bleu_floor"].items()}
        self.gt_labels = gt_labels
        self.gt_reports = gt_reports
        self.alpha, self.gamma, self.lambda_bleu = alpha, gamma, lambda_bleu
        self.last_terms: List[Dict[str, float]] = []
        self.n_scored = 0

    # -- the two metric terms -------------------------------------------------
    def _s_i(self, pred: Sequence[int], gt: Sequence[int]) -> float:
        tp = sum(1 for j in range(N_LABELS) if pred[j] and gt[j])
        fn = sum(1 for j in range(N_LABELS) if not pred[j] and gt[j])
        fp = sum(1 for j in range(N_LABELS) if pred[j] and not gt[j])
        return self.r * tp - self.r * fn - fp

    def _d_f1(self, pred: Sequence[int], gt: Sequence[int]) -> float:
        """macroF1(frozen corpus + this rollout) - macroF1(frozen corpus).

        The corpus is never updated, so every rollout of every step is measured
        against an identical context. That is both what makes a GRPO group's
        advantages comparable and what keeps the term's magnitude constant.
        """
        dtp, dfp, dfn = _classify_counts(pred, gt)
        return macro_f1([self.tp[j] + dtp[j] for j in range(N_LABELS)],
                        [self.fp[j] + dfp[j] for j in range(N_LABELS)],
                        [self.fn[j] + dfn[j] for j in range(N_LABELS)]) - self.base_f1

    # -- TRL entry point ------------------------------------------------------
    def __call__(self, prompts, completions, sample_name=None, log_metric=None,
                 **kwargs) -> List[float]:
        if sample_name is None:
            raise ValueError(
                "the dataset must carry a `sample_name` column: the reward needs it "
                "for the ground-truth labels, the GT report and the BLEU floor.")
        texts = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
        if len(texts) != len(sample_name):
            raise ValueError(f"{len(texts)} completions vs {len(sample_name)} names")

        preds = self.predict_labels(texts)
        refs = []
        for name in sample_name:
            if name not in self.gt_reports:
                raise KeyError(f"{name} has no ground-truth report")
            if name not in self.bleu_floor:
                raise KeyError(
                    f"{name} has no BLEU floor. The reference generation did not "
                    f"cover the RL subset -- rebuild it over subset8000.json.")
            refs.append(self.gt_reports[name])
        bleus = bleu_mean_per_item(refs, texts)

        out, terms = [], []
        for pred, name, bleu in zip(preds, sample_name, bleus):
            gt = self.gt_labels.get(name)
            if gt is None:
                raise KeyError(f"{name} is absent from the organiser label file")
            crg_t = self.alpha * self._s_i(pred, gt) / self.sigma_s
            f1_t = self.gamma * self._d_f1(pred, gt) / self.sigma_f1
            pen = self.lambda_bleu * max(0.0, self.bleu_floor[name] - bleu) / self.sigma_bleu
            out.append(crg_t + f1_t - pen)
            terms.append({"crg": crg_t, "f1": f1_t, "bleu_pen": pen, "bleu": bleu})
        self.last_terms = terms
        self.n_scored += len(terms)

        # The 50-step gate reads these. The pilot logged only the sum, sat at
        # -0.85 for 500 steps, and there was no way to tell whether the F1 term
        # was doing anything or the BLEU penalty was simply always on. The three
        # std's are the check that "one vote per term" actually holds: each
        # should be ~1 by construction of the normalisers.
        if log_metric is not None and terms:
            for key in ("crg", "f1", "bleu_pen", "bleu"):
                vals = np.array([t[key] for t in terms], dtype=float)
                log_metric(f"reward_{key}", float(vals.mean()))
                log_metric(f"reward_{key}_std", float(vals.std()))
            log_metric("reward_bleu_pen_frac",
                       float(np.mean([t["bleu_pen"] > 0 for t in terms])))
        return out

    # -- RadBERT --------------------------------------------------------------
    def predict_labels(self, texts: Sequence[str]) -> List[List[int]]:
        raise NotImplementedError("attach a labeller with `with_radbert()`")

    def with_radbert(self, radbert_dir: str = (
                         "/data/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/"
                         "reportgen_evaluation/assets/radbert_local"),
                     ckpt: str = "/data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth",
                     device: str = "cuda", threshold: float = 0.5,
                     batch_size: int = 32) -> "CRGF1BLEUReward":
        """The scorer's own RadBertClassifier -- same checkpoint, same 0.5
        threshold, same max_length 512 that reportgen_evaluation/src/infer.py
        uses. Both sides of macro-F1 and CRG are RadBERT-defined: the GT csv
        reproduces from this classifier on the GT reports at 99.61% cell
        agreement. The judge is a text classifier, not the image.
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
