"""Adapt the GT sentence labels into the token-weight loader's schema.

    uv run --no-project --python .venv/bin/python python scripts/build_gt_sentence_weights.py \
        --split_json /data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_train_filtered_prob_half.json \
        --gt_spans     GT_report_sentence_label/train_sentence_token_labels.json \
        --gt_sentences GT_report_sentence_label/train_sentences_18labels.json \
        --out          data/sentence_labels/train_gt_sentence_labels.json

WHAT THIS IS FOR
----------------
``GT_report_sentence_label/*_sentence_token_labels.json`` already carries
exactly the structure ``CTRATEDataset_Train`` reads --
``{key: {"spans": [[start, end, has_label], ...]}}`` -- so nothing in the
training path has to change. Two things do not line up, and this script is
where they are fixed, once, offline:

1. KEY NAMESPACE. The GT files are keyed per study (``train_10000_a``) while
   the split's ``sample_name`` carries the reconstruction index
   (``train_10000_a_1``). Every sample is remapped to its own name, so the
   dataset's lookup stays a plain ``sample_name`` hit and no key-munging logic
   leaks into ``__getitem__``.

2. COVERAGE. 19 train and 1 valid ``sample_name`` have no GT entry at all.
   They are written to ``<out>.missing.txt`` and this script exits 3, because a
   labels file that covers 99.92% of the corpus is a run at a silently
   different objective for the rest. The dataset refuses such a file at
   construction anyway; this just surfaces it before a job is queued.

   ``--filtered_split_out`` is the resolution chosen for this experiment: it
   writes the split with those samples removed, so the run matches the label
   author's N exactly (23,850 train / 1,551 valid) instead of leaving 0.08% of
   the corpus at a different objective. The arm config points
   ``train_json_path`` / ``valid_json_path`` at the result, which keeps the
   whole decision in configs and data files -- ``src/`` is untouched. The cost
   is that the arm's corpus is 0.08% smaller than the baseline's, so this is
   not a perfectly single-factor arm; it is an accepted, documented deviation
   for this experiment, not the pattern to reach for again. Ask the label
   author for the missing 20 before a run that has to be strictly comparable.

This script does NOT relabel anything. It is a pure remap plus verification, so
the weights stay traceable to a single label source -- mixing in a second
labeller for the missing samples would make our numbers non-comparable with the
w=5 result this arm exists to reproduce.

WHY THE VERIFICATION IS NOT OPTIONAL
------------------------------------
The spans are raw character offsets into ``report_metadata``. If they were
measured against a differently-preprocessed report string, every weight lands
on the wrong words and the loss curve looks perfectly normal while doing it. So
every span is checked against the actual training target here: in bounds,
monotonic, and ``report_metadata[start:end]`` reproducing the stored sentence.

Two benign differences are tolerated and counted, never silently:

  * whitespace -- the stored sentences have newlines normalised to spaces
    (93 sentences on train), which cannot move an offset;
  * ``nan`` -- 16 train sentences store ``"Impression: nan"`` where the report
    reads ``"Impression: Not"`` and similar, an artefact of the upstream
    pandas read.

Both keep the span length identical, which is what actually guarantees the
offsets are usable. Any difference in LENGTH raises: that means the spans were
cut from a different string than the one we train on.

``has_label`` is also re-derived from the 18-dimensional per-sentence vectors in
``--gt_sentences`` and asserted to equal the flag in ``--gt_spans``, so a
mismatch between the two files cannot pass unnoticed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# sample_name -> study key: strip one trailing reconstruction index. Anchored so
# that a name without one is left alone rather than losing a real suffix.
_RECON_SUFFIX = re.compile(r"_\d+$")
_WS = re.compile(r"\s+")

EXIT_INCOMPLETE = 3


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def load_split(path: str) -> "list[tuple[str, dict]]":
    """The split files are JSON Lines, one record per volume.

    The raw line is kept alongside the parsed record so that
    ``--filtered_split_out`` can copy surviving records through byte for byte
    rather than re-serialising them (which would silently reformat floats and
    key order in a file the training run reads).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append((line, json.loads(line)))
    if not rows:
        raise ValueError(f"{path} holds no records")
    return rows


def gt_key(sample_name: str, gt: dict) -> "str | None":
    """The GT key for a sample, or None if it has no entry.

    Tried in order: the name itself (in case a future GT file is already keyed
    by sample_name), then the name with one reconstruction suffix removed.
    """
    if sample_name in gt:
        return sample_name
    stripped = _RECON_SUFFIX.sub("", sample_name)
    if stripped != sample_name and stripped in gt:
        return stripped
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_json", required=True,
                    help="the split the training run reads (JSON Lines)")
    ap.add_argument("--gt_spans", required=True,
                    help="GT_report_sentence_label/*_sentence_token_labels.json")
    ap.add_argument("--gt_sentences", required=True,
                    help="GT_report_sentence_label/*_sentences_18labels.json, used to "
                         "verify the spans and to re-derive has_label")
    ap.add_argument("--out", required=True,
                    help="written keyed by sample_name; feed this to "
                         "sentence_labels_path / valid_sentence_labels_path")
    ap.add_argument("--filtered_split_out", default=None,
                    help="if given, write the split with the samples that have no "
                         "GT entry removed, so the run's N matches the label "
                         "author's exactly. Point train_json_path / "
                         "valid_json_path at this file. Lines are copied verbatim.")
    args = ap.parse_args()

    rows = load_split(args.split_json)
    records = [rec for _, rec in rows]
    with open(args.gt_spans, encoding="utf-8") as f:
        gt_spans = json.load(f)
    with open(args.gt_sentences, encoding="utf-8") as f:
        gt_sents = json.load(f)

    if set(gt_spans) != set(gt_sents):
        raise ValueError(
            f"--gt_spans and --gt_sentences describe different sample sets "
            f"({len(gt_spans)} vs {len(gt_sents)} keys); they must be two views "
            f"of the same labelling run")

    out: dict = {}
    missing: "list[str]" = []
    n_sent = n_fire = 0
    chars_total = chars_fire = 0
    n_ws_diff = n_nan_diff = 0

    for rec in records:
        name = rec["sample_name"]
        report = rec["report_metadata"]
        key = gt_key(name, gt_spans)
        if key is None:
            missing.append(name)
            continue

        spans = gt_spans[key]["spans"]
        sents = gt_sents[key]["sentences"]
        labels = gt_sents[key]["labels"]
        if not (len(spans) == len(sents) == len(labels)):
            raise ValueError(
                f"{name} ({key}): {len(spans)} spans but {len(sents)} sentences "
                f"and {len(labels)} label vectors")
        declared = gt_spans[key].get("n")
        if declared is not None and int(declared) != len(spans):
            raise ValueError(f"{name} ({key}): n={declared} but {len(spans)} spans")

        spans_out = []
        prev_end = 0
        for i, span in enumerate(spans):
            start, end, flag = int(span[0]), int(span[1]), int(bool(span[2]))
            if not 0 <= start < end <= len(report):
                raise ValueError(
                    f"{name} ({key}) span {i} = [{start}, {end}) is outside "
                    f"report_metadata of length {len(report)}; the spans were "
                    f"measured against a different string")
            if start < prev_end:
                raise ValueError(
                    f"{name} ({key}) span {i} starts at {start}, before the "
                    f"previous span ended at {prev_end}; spans must be disjoint "
                    f"and in order")
            prev_end = end

            actual, stored = report[start:end], sents[i]
            if actual != stored:
                if len(actual) != len(stored):
                    raise ValueError(
                        f"{name} ({key}) span {i}: report_metadata[{start}:{end}] "
                        f"is {len(actual)} chars but the stored sentence is "
                        f"{len(stored)}; the offsets do not index this report.\n"
                        f"  report: {actual!r}\n  stored: {stored!r}")
                # Same length, so the offsets are still exact. Only the stored
                # copy of the text differs -- see the module docstring.
                if _norm(actual) == _norm(stored):
                    n_ws_diff += 1
                else:
                    n_nan_diff += 1

            # has_label must agree with the 18-label vector it came from.
            from_vector = int(any(int(v) for v in labels[i]))
            if flag != from_vector:
                raise ValueError(
                    f"{name} ({key}) span {i}: has_label={flag} but the 18-label "
                    f"vector gives {from_vector}; --gt_spans and --gt_sentences "
                    f"disagree")

            spans_out.append([start, end, flag])
            n_sent += 1
            n_fire += flag
            chars_total += end - start
            chars_fire += (end - start) if flag else 0

        out[name] = {"spans": spans_out}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[out] {out_path}  ({len(out)} samples keyed by sample_name)", flush=True)

    stats = {
        "split_json": args.split_json,
        "gt_spans": args.gt_spans,
        "gt_sentences": args.gt_sentences,
        "n_split_samples": len(records),
        "n_gt_keys": len(gt_spans),
        "n_written": len(out),
        "n_missing": len(missing),
        "filtered_split_out": args.filtered_split_out,
        "n_sentences": n_sent,
        "sentence_fire_rate": n_fire / max(1, n_sent),
        "char_fire_rate": chars_fire / max(1, chars_total),
        "n_whitespace_only_text_diff": n_ws_diff,
        "n_equal_length_text_diff": n_nan_diff,
    }
    stats_path = str(out_path.with_suffix("")) + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"sentences               : {n_sent}")
    print(f"sentences firing        : {stats['sentence_fire_rate']:6.2%}")
    print(f"report characters firing: {stats['char_fire_rate']:6.2%}"
          "   <- the share a multiplier acts on (token share is measured by "
          "scripts/audit_token_weight.py)")
    print(f"text diffs tolerated    : {n_ws_diff} whitespace-only, "
          f"{n_nan_diff} same-length other (offsets unaffected)")
    print(f"[stats] {stats_path}", flush=True)

    if missing:
        missing_path = str(out_path) + ".missing.txt"
        with open(missing_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")
        print(f"\n{'=' * 70}")
        print(f"INCOMPLETE: {len(missing)} of {len(records)} samples have no GT entry")
        print(f"  {missing_path}")

    if args.filtered_split_out:
        drop = set(missing)
        split_path = Path(args.filtered_split_out)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with open(split_path, "w", encoding="utf-8") as f:
            for line, rec in rows:
                if rec["sample_name"] in drop:
                    continue
                f.write(line if line.endswith("\n") else line + "\n")
                kept += 1
        # The point of the filtered split is that the dataset's coverage check
        # passes without weakening it, so the two counts must agree exactly.
        if kept != len(out):
            raise RuntimeError(
                f"wrote {kept} records but the labels file covers {len(out)} "
                f"samples; the filtered split and the weights disagree")
        print(f"[filtered split] {split_path}  ({kept} records, "
              f"{len(drop)} dropped to match the GT labelling's N)")
        print("  Point train_json_path / valid_json_path at this file. The arm's "
              "corpus is\n  now smaller than the baseline's by those dropped "
              "records -- an accepted\n  deviation for this experiment; note it "
              "when reporting the comparison.", flush=True)
    elif missing:
        print("  Ask the label author for these, or pass --filtered_split_out to")
        print("  drop them and match the labelling's N. CTRATEDataset_Train")
        print("  refuses a partial labels file at construction, so training")
        print("  cannot start until one of the two is done.")
        print(f"{'=' * 70}", flush=True)
        sys.exit(EXIT_INCOMPLETE)


if __name__ == "__main__":
    main()
