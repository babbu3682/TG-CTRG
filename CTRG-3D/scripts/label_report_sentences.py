"""FALLBACK sentence labeller. Prefer the colleague's own spans file.

The primary source of ``{sample_name: {"spans": [[start, end, has_label], ...]}}``
is the refiner3d author, who produced these for the runs this idea comes from.
``refiner3d_share/tools/label_sentences.py`` and the ``*_sentence_labels.json``
samples arrived at 0 bytes, so ask for them rather than assuming this script's
output is equivalent -- a differently-labelled corpus quietly makes our numbers
non-comparable with theirs, which is the whole reason for reproducing w=5.

This script exists only for the case where those files cannot be obtained, or
where a new split has to be labelled from scratch.

    uv run --no-project --python .venv/bin/python python scripts/label_report_sentences.py \
        --split_json /data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_train_filtered_prob_half.json \
        --out        data/sentence_labels/train_sentence_labels.json

WHY THE SCORER'S CLASSIFIER AND NOT THE SPLIT'S OWN FIELDS
----------------------------------------------------------
``merged_*_filtered_prob_half.json`` carries ``sentences`` / ``labels``
/ ``sentence_probs``, but those sentences are PhraseAlign-normalised key phrases
("Patent trachea and both main bronchi"), not spans of ``report_metadata``
("Trachea, both main bronchi are open."). They are reordered and reworded, so
they cannot be mapped back to character offsets in the training target. This
script splits ``report_metadata`` itself and classifies each of its sentences,
so the weights land on exactly the characters the model is trained to emit.

Using ``RadBertClassifier.pth`` -- the same checkpoint
``src/evaluate_reportgen.py`` scores with -- is the point: training weight then
sits on precisely the text the evaluator reads back out. A sentence classified
with anything else would be a proxy for the metric rather than the metric.

CAVEAT, stated because it is real and NOT currently checked: the classifier was
trained on whole reports (max_length 512), so feeding it one sentence at a time
is out of distribution, and nothing here verifies the decomposition. The split's
``volume_label`` must NOT be used as the reference for such a check -- it is not
accurate, and precision/recall computed against it would look like evidence
while guaranteeing nothing. If the spans need validating, agree on a trustworthy
reference first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

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

DEFAULT_RADBERT_CKPT = "/data/MIDS/datasets/CT-RATE/models/RadBertClassifier.pth"
DEFAULT_RADBERT_DIR = (
    "/data/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/"
    "reportgen_evaluation/assets/radbert_local"
)

# Sentence terminator only when whitespace (or end of string) follows, so that
# "1.5 cm", "15 mm.)" and "(chronic occlusion pathology?)." do not split.
_TERMINATOR = re.compile(r"[.!?]+(?=\s|$)")
_NEWLINE = re.compile(r"\n+")
# CT-RATE reports are "Findings: ... \nImpression: ...". The header is cut off
# the following sentence so that the literal word "Findings:" never ends up
# inside an upweighted span.
_HEADER = re.compile(r"\b(?:Findings|Impressions?|Note)\s*:")

MIN_SPAN_CHARS = 3


def split_sentences(text: str) -> list[tuple[int, int]]:
    """``[(start, end)]`` character spans of `text`, whitespace trimmed.

    Spans are returned in reading order and never overlap, but they do not
    tile the string: section headers and inter-sentence whitespace fall
    between them and so are left at weight 1.0 by construction.
    """
    cuts = {0, len(text)}
    for rx in (_TERMINATOR, _HEADER):
        for m in rx.finditer(text):
            cuts.add(m.end())
    for m in _NEWLINE.finditer(text):
        cuts.add(m.start())
        cuts.add(m.end())

    spans = []
    ordered = sorted(cuts)
    for a, z in zip(ordered, ordered[1:]):
        seg = text[a:z]
        if not seg.strip():
            continue
        start = a + (len(seg) - len(seg.lstrip()))
        end = z - (len(seg) - len(seg.rstrip()))
        if end - start >= MIN_SPAN_CHARS:
            spans.append((start, end))
    return spans


def load_classifier(radbert_dir: str, ckpt_path: str, device: torch.device):
    """The scorer's ``RadBertClassifier``, imported from its own source tree."""
    repo = Path(os.environ.get(
        "REPORTGEN_EVAL_DIR",
        "/data/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_evaluation",
    ))
    src = repo / "src"
    if not src.is_dir():
        sys.exit(f"reportgen_evaluation/src not found at {src}; set $REPORTGEN_EVAL_DIR")
    sys.path.insert(0, str(src))

    # classifier.py hardcodes MODEL_DIR = /opt/app/radbert_local (the Docker
    # image's asset path); override it before RadBertClassifier() is built, the
    # same way src/evaluate_reportgen.py does.
    import classifier as classifier_mod
    classifier_mod.MODEL_DIR = Path(radbert_dir).resolve()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(radbert_dir, do_lower_case=True)

    model = classifier_mod.RadBertClassifier(n_classes=len(LABEL_COLS))
    state = torch.load(ckpt_path, map_location="cpu")
    # Older transformers saved `position_ids` as a persistent buffer; newer
    # versions removed it. infer.py strips it the same way.
    state = {k: v for k, v in state.items() if not k.endswith("position_ids")}
    model.load_state_dict(state, strict=True)
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def classify(sentences: list[str], tokenizer, model, device, batch_size: int) -> np.ndarray:
    """``[N, 18]`` sigmoid probabilities, one row per sentence."""
    out = np.zeros((len(sentences), len(LABEL_COLS)), dtype=np.float32)
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        # padding=True, not the scorer's padding='max_length': RoBERTa derives
        # its position ids from the padding mask, so padded positions shift
        # nothing and the two are numerically identical -- this one is ~10x
        # faster on sentence-length inputs.
        enc = tokenizer(chunk, return_tensors="pt", max_length=512,
                        padding=True, truncation=True)
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        out[i:i + len(chunk)] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_json", required=True,
                    help="merged_{train,valid}_filtered_prob_half.json (JSONL).")
    ap.add_argument("--out", required=True, help="Destination JSON of spans.")
    ap.add_argument("--radbert_model_path", default=DEFAULT_RADBERT_CKPT)
    ap.add_argument("--radbert_dir", default=DEFAULT_RADBERT_DIR)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold; 0.5 is what the scorer's ModelTrainer uses.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="Label only the first N volumes (for a quick check).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    records = []
    with open(args.split_json, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if args.limit and len(records) >= args.limit:
                break
    print(f"[data] {len(records)} volumes from {args.split_json}", flush=True)

    # One flat list of sentences across the whole split, so the classifier runs
    # in full batches instead of one ragged batch per report.
    flat, owner = [], []
    per_sample: dict = {}
    for rec in records:
        name = rec["sample_name"]
        report = rec["report_metadata"]
        spans = split_sentences(report)
        per_sample[name] = {"spans": spans}
        for (a, b) in spans:
            flat.append(report[a:b])
            owner.append((name, len(flat) - 1))
    print(f"[data] {len(flat)} sentences "
          f"({len(flat) / max(1, len(records)):.1f} per volume)", flush=True)

    tokenizer, model = load_classifier(args.radbert_dir, args.radbert_model_path, device)
    probs = classify(flat, tokenizer, model, device, args.batch_size)
    hits = (probs >= args.threshold)

    out: dict = {}
    n_fire = 0
    chars_total = chars_fire = 0
    cursor = 0
    for rec in records:
        name = rec["sample_name"]
        entry = per_sample[name]
        spans_out = []
        for (a, b) in entry["spans"]:
            flag = bool(hits[cursor].any())
            # The max probability over the 18 labels is carried through so that
            # a later run can re-threshold or soften the weights without
            # re-running the classifier over 24k reports.
            spans_out.append([int(a), int(b), int(flag), round(float(probs[cursor].max()), 4)])
            n_fire += int(flag)
            chars_total += b - a
            chars_fire += (b - a) if flag else 0
            cursor += 1
        out[name] = {"spans": spans_out}
    assert cursor == len(flat)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[out] {args.out}", flush=True)

    stats = {
        "split_json": args.split_json,
        "n_volumes": len(records),
        "n_sentences": len(flat),
        "threshold": args.threshold,
        "sentence_fire_rate": n_fire / max(1, len(flat)),
        "char_fire_rate": chars_fire / max(1, chars_total),
    }
    stats_path = str(Path(args.out).with_suffix("")) + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nsentences firing        : {stats['sentence_fire_rate']:6.2%}")
    print(f"report characters firing: {stats['char_fire_rate']:6.2%}"
          "   <- the share a multiplier acts on (token share is close but measured by scripts/audit_token_weight.py)")
    print(f"\n[stats] {stats_path}", flush=True)


if __name__ == "__main__":
    main()
