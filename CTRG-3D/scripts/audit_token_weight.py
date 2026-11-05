"""Verify the token-weighted loss before spending GPU-days on it.

    uv run --no-project --python .venv/bin/python python scripts/audit_token_weight.py \
        --split_json      /data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_train_filtered_prob_half.json \
        --rve_manifest    /data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/train/mapping_path.csv \
        --sentence_labels data/sentence_labels/train_sentence_labels.json \
        --tokenizer_path  <Qwen3-8B> --n 200

Three checks, in the order that matters:

1. IDENTITY -- with all-ones weights, ``weighted_causal_lm_loss`` reproduces
   transformers' own ``ForCausalLMLoss`` to float precision. A feature that can
   be switched off must not change behaviour when it is off.

2. ALIGNMENT -- for real samples, decode exactly the token positions that got
   the raised weight and compare them, character by character, against the
   sentences scripts/label_report_sentences.py marked. This is a position-level
   audit, not a spot check: an off-by-one in the offset mapping would train the
   model to emphasise the wrong words while every loss curve looked perfectly
   normal, so the run must not start until the recovered text matches.

3. CALIBRATION -- measure what share of supervised tokens are label-bearing,
   and print the loss share each candidate weight buys. Choose the weight from
   the share you want, not the other way round; the share is the quantity that
   actually describes the objective, and it shifts whenever the templates, the
   prompt or the split change.

The CT volume is never read: ``load_rve_volume`` is stubbed out so the real
text path runs with no I/O. Everything audited here is text.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))


def check_identity(device: str) -> None:
    """All-ones weights must reproduce transformers' ForCausalLMLoss exactly."""
    from transformers.loss.loss_utils import ForCausalLMLoss
    from Model.CTRG_3D import weighted_causal_lm_loss

    torch.manual_seed(0)
    B, L, V = 3, 64, 512
    logits = torch.randn(B, L, V, device=device)
    labels = torch.randint(0, V, (B, L), device=device)
    # A realistic mask: a prompt prefix and a padded tail are both ignored.
    labels[:, :20] = -100
    labels[1, 50:] = -100

    ours = weighted_causal_lm_loss(logits, labels, torch.ones_like(labels, dtype=torch.float32))
    theirs = ForCausalLMLoss(logits, labels, vocab_size=V)
    delta = float((ours - theirs).abs())
    print(f"[1/3 identity] weighted(w=1) {ours:.10f}   ForCausalLMLoss {theirs:.10f}   |diff| {delta:.2e}")
    assert delta < 1e-5, "all-ones weighting does NOT reproduce the uniform loss"

    # And a raised weight must actually move it, or the plumbing is dead.
    w = torch.ones_like(labels, dtype=torch.float32)
    w[:, 30:40] = 5.0
    moved = weighted_causal_lm_loss(logits, labels, w)
    assert float((moved - ours).abs()) > 1e-4, "raising the weight changed nothing"
    print(f"[1/3 identity] weighted(w=5 on a slice) {moved:.10f}  (differs, as it must)")


def build_dataset(args):
    """The real CTRATEDataset_Train with the volume loader stubbed out."""
    import Dataset.ctrate_dataset_train as ds_mod

    def _stub(filepath, target_slices=None, target_h=None, target_w=None,
              pad_value=-1024.0, crop_mode="center"):
        return torch.zeros(1, 8, 8, 8)

    ds_mod.load_rve_volume = _stub

    return ds_mod.CTRATEDataset_Train(
        tokenizer_path=args.tokenizer_path,
        image_folder="", report_folder="",
        json_path=args.split_json,
        rve_manifest_path=args.rve_manifest,
        max_seq=args.max_seq,
        image_num=args.num_visual_tokens,
        use_fast=False,
        use_cls_template=bool(args.cls_template_path),
        use_ret_template=bool(args.ret_template_path),
        cls_template_path=args.cls_template_path,
        ret_template_path=args.ret_template_path,
        # Dropout off: the audit is about token positions, and a randomly
        # shortened prompt only makes the printed comparison harder to read.
        # It cannot affect alignment either way -- the spans are located by
        # searching for the answer inside the final text, whatever precedes it.
        is_train=False,
        sentence_labels_path=args.sentence_labels,
        label_token_weight=args.weight,
    )


def check_alignment(ds, n: int, show: int) -> tuple[int, int]:
    """Decode the upweighted positions and compare them with the marked spans."""
    import json as _json

    tok = ds.tokenizer
    n_sup_total = n_hot_total = 0
    n_mismatch = 0
    mismatches = []

    for i in range(min(n, len(ds))):
        sample = ds[i]
        name = ds.df.iloc[i].get("sample_name", str(i))
        label, w = sample["label"], sample["token_weight"]
        ids = sample["lang_x"]

        supervised = label != -100
        hot = supervised & (w > 1.0)
        n_sup_total += int(supervised.sum())
        n_hot_total += int(hot.sum())

        # What the weights actually cover, read straight off the token ids.
        recovered = tok.decode(ids[hot], skip_special_tokens=True)
        # What they were supposed to cover. A gap between two CONSECUTIVE
        # flagged spans belongs to no sentence at all, so its characters are
        # legitimately upweighted (the token straddling the boundary overlaps a
        # flagged span and is weighted whole) and must be part of the
        # expectation. Usually that gap is the space between two sentences,
        # which the alphanumeric comparison below ignores anyway -- but the GT
        # labelling drops 2 characters mid-word in one train report
        # (train_4677_a_1: "hiatal he|rn|ia"), and without this the audit
        # reports a misalignment that is not one.
        report = ds.df.iloc[i]["report_metadata"]
        spans = ds.sentence_labels[name]["spans"]
        pieces = []
        for k, span in enumerate(spans):
            s, e, flag = span[0], span[1], span[2]
            if not flag:
                continue
            pieces.append(report[s:e])
            nxt = spans[k + 1] if k + 1 < len(spans) else None
            if nxt is not None and nxt[2] and nxt[0] > e:
                pieces.append(report[e:nxt[0]])
        expected = "".join(pieces)

        # Compare on alphanumerics only: the recovered string is a concatenation
        # of non-adjacent token runs, so inter-sentence spacing differs by
        # construction and is not evidence of misalignment.
        r_key = "".join(c for c in recovered.lower() if c.isalnum())
        e_key = "".join(c for c in expected.lower() if c.isalnum())
        if r_key != e_key:
            n_mismatch += 1
            # Bounded: over a full split the reports alone would be hundreds of
            # MB, and three examples are enough to diagnose an offset bug.
            if len(mismatches) < 3:
                mismatches.append((name, expected, recovered))

        if i < show:
            print(f"\n--- {name}  ({int(hot.sum())}/{int(supervised.sum())} supervised tokens upweighted)")
            print(f"    marked   : {expected[:400]!r}")
            print(f"    recovered: {recovered[:400]!r}")

    print(f"\n[2/3 alignment] {min(n, len(ds))} samples, {n_mismatch} mismatched")
    for name, exp, rec in mismatches:
        print(f"  MISMATCH {name}\n    marked   : {exp[:300]!r}\n    recovered: {rec[:300]!r}")
    assert not n_mismatch, (
        f"{n_mismatch} samples have token weights that do not cover the marked "
        f"sentences -- do NOT start training on these spans")
    return n_hot_total, n_sup_total


def report_calibration(n_hot: int, n_sup: int) -> None:
    f = n_hot / max(1, n_sup)
    print(f"\n[3/3 calibration] label-bearing share of supervised tokens: {f:.2%} "
          f"({n_hot:,} of {n_sup:,})")
    print("\n  Pick the weight from the loss share you want, not by feel:")
    print(f"  {'weight w':>9}  {'loss share of label tokens':>28}")
    for w in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0):
        share = w * f / (w * f + (1 - f))
        print(f"  {w:>9.1f}  {share:>27.1%}")
    print("\n  Inverse, for a target share a:  w = a(1-f) / ((1-a)f)")
    for a in (0.5, 0.6, 0.7, 0.8, 0.9):
        w = a * (1 - f) / ((1 - a) * f)
        print(f"    share {a:.0%}  ->  label_token_weight = {w:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_json")
    ap.add_argument("--rve_manifest")
    ap.add_argument("--sentence_labels")
    ap.add_argument("--tokenizer_path")
    ap.add_argument("--cls_template_path", default=None)
    ap.add_argument("--ret_template_path", default=None)
    ap.add_argument("--max_seq", type=int, default=2688)
    ap.add_argument("--num_visual_tokens", type=int, default=128)
    ap.add_argument("--weight", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=200, help="Samples to audit.")
    ap.add_argument("--show", type=int, default=2, help="Samples to print in full.")
    ap.add_argument("--identity_only", action="store_true",
                    help="Run check 1 only; needs no data or tokenizer.")
    args = ap.parse_args()

    check_identity("cuda" if torch.cuda.is_available() else "cpu")
    if args.identity_only:
        return

    missing = [k for k in ("split_json", "rve_manifest", "sentence_labels", "tokenizer_path")
               if getattr(args, k) is None]
    if missing:
        sys.exit(f"checks 2 and 3 need: {', '.join('--' + m for m in missing)}")

    ds = build_dataset(args)
    n_hot, n_sup = check_alignment(ds, args.n, args.show)
    report_calibration(n_hot, n_sup)
    print("\nOK -- weights are aligned and the identity holds.")


if __name__ == "__main__":
    main()
