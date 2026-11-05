"""abn classification probe -> per-sample cls_report template entry (lapa).

Reproduces pillar-finetune's ``CTRATE_ABN_cls_report_templates_lapa_f1max``
— the template set the newenc-B SFT and every GRPO draw were prompted with
(``generate_batched.py`` SPLIT_PATHS["valid_newenc"]["cls"]).

RECIPE, ESTABLISHED BY REPRODUCTION (2026-08-30) — do not substitute
-------------------------------------------------------------------
    feature    = encoder ``pooled``      (384-d, already L2-normalised)
    head       = lapa_lz4_half/finetune_head.pt
                 Linear(384,1024) -> ReLU -> Dropout -> Linear(1024,18)
    threshold  = data/abn_thresholds_f1max_train_lapa.json, PER CLASS,
                 positive only: p >= t -> positive, else negative
    sentences  = assets/abnclass_config.yaml report_templates

Checks that pinned each choice, against ``cls_report_dev.json`` (1,552 exams):
  * ``finetune_head.pt`` reproduces ``prob_*`` to max|delta| 1e-6; the next best
    candidate in the same directory is off by 0.351.
  * the training cache's ``features`` column is ``torch.equal`` to ``pooled``,
    and ``||pooled|| == 1.0000`` — so no extra normalisation belongs here.
  * the per-class rule reproduces all 18x1,552 ``status_*`` fields exactly, and
    the config's sentences reproduce all 1,552 ``report`` lists exactly.

There is NO "uncertain" band: every one of the 18 labels emits a sentence.
That is why the p90/n20 knobs of the previous (open-source-encoder) container
are gone rather than merely retuned — a different rule, not a different value.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Label order == prediction dim order (pillar/metrics/abn_metric.py).
LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DEFAULT_TEMPLATE_CONFIG = ASSETS_DIR / "abnclass_config.yaml"

PROBE_HEAD_NAME = "cls_finetune_head.pt"
THRESHOLDS_NAME = "abn_thresholds_f1max_train_lapa.json"

IN_DIM = 384


def load_sentences(config_path: str | Path = DEFAULT_TEMPLATE_CONFIG):
    """LABEL_COLS-ordered [(positive_sentence, negative_sentence), ...]."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    rt = cfg.get("report_templates")
    if rt is None:
        raise ValueError(f"{config_path} has no report_templates section")
    by_name = {e["name"]: e for e in rt["labels"]}
    missing = [c for c in LABEL_COLS if c not in by_name]
    if missing:
        raise ValueError(f"report_templates missing labels: {missing}")
    return [(by_name[c]["positive"], by_name[c]["negative"]) for c in LABEL_COLS]


def load_probe_head(probe_head_path: str | Path, hidden: int = 1024,
                    in_dim: int = IN_DIM, n_labels: int = len(LABEL_COLS)):
    """train_abn_probe.py's layout; strict load verifies it. Dropout is identity."""
    state = torch.load(probe_head_path, map_location="cpu", weights_only=False)
    head = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(hidden, n_labels),
    )
    head.load_state_dict(state, strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def load_thresholds(path: str | Path) -> list[float]:
    """LABEL_COLS-ordered per-class positive thresholds."""
    with open(path) as f:
        blob = json.load(f)
    pos = blob["positive"]
    missing = [c for c in LABEL_COLS if c not in pos]
    if missing:
        raise ValueError(f"{path}: no threshold for {missing}")
    return [float(pos[c]) for c in LABEL_COLS]


@torch.no_grad()
def build_cls_entry(head, sentences, thresholds: list[float],
                    pooled: torch.Tensor, sample_name: str) -> dict:
    """pooled: (1, 384) fp32 CPU tensor -> cls_report json row.

    Schema identical to build_cls_report_templates.py:
        {"accession", "report": [18 sentences],
         "status_<label>": positive|negative, "prob_<label>": float}
    """
    if pooled.shape[-1] != IN_DIM:
        raise ValueError(f"cls probe expects {IN_DIM}-d pooled, got {tuple(pooled.shape)}")
    probs = torch.sigmoid(head(pooled.float()))[0]

    row = OrderedDict(accession=str(sample_name), report=[])
    for i, label in enumerate(LABEL_COLS):
        p = float(probs[i])
        positive = p >= thresholds[i]
        row["report"].append(sentences[i][0] if positive else sentences[i][1])
        row[f"status_{label}"] = "positive" if positive else "negative"
    for i, label in enumerate(LABEL_COLS):
        row[f"prob_{label}"] = round(float(probs[i]), 6)
    return row
