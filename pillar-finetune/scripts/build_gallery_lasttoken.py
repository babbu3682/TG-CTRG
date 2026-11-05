#!/usr/bin/env python3
"""
Re-embed the retrieval gallery with last-token pooling.

data/gallery_embeddings.pt (79,166 sentences, used by RetrievalMetric's recall@k-style
gallery matching) was built 2026-07-27, before the project switched the text cache from
mean-pooling to last-token pooling on 2026-08-07 (see CLAUDE.md
"text_cache_qwen3_embedding_8b_phrase의 max_length가 512로 축소되어 있음"). Verified
directly by re-encoding 5 gallery sentences both ways and comparing cosine similarity
against the stored vectors: they match mean-pooling (cos up to 0.9998) and do NOT match
last-token pooling (cos as low as 0.22) -- so gallery_embeddings.pt is a mean-pool cache.

This means the "finetune_ret_lasthidden" retrieval run (job 26978948, image tower trained
to align with last-token-pooled text) has been evaluated against a gallery in a different
embedding space -- RetrievalMetric's recall@k/rank numbers for that job are not trustworthy
until this exists. (val_loss is unaffected: RetrievalLoss never touches the gallery, and
main.monitor: val_loss is what drives early stopping/checkpoint selection -- confirmed by
grepping pillar/losses/retrieval_loss.py for "gallery": no hits.)

Sentences and their order are NOT regenerated (gallery_sentences.pt is reused as-is) -- only
the embeddings are recomputed, with the same recipe already verified against the pretrain
text cache (CachedTextEmbedding._encode_texts): left-padding, last_hidden_state[:, -1],
L2-normalized. max_length=256, matching key_phrase_max_length in the original (correct)
pretrain cache build -- these gallery sentences are single findings ("Patent trachea and
both main bronchi"), the same kind of short report-derived sentence as key_phrase, not full
reports, and measured well under that cap (max 65 words in the existing sentence file).

Output: data/gallery_embeddings_lasthidden.pt -- a NEW file, gallery_embeddings.pt (mean-pool)
is left untouched since the finetune_ret (job 26947705, mean-pool) run still needs it.

Usage:
    uv run python scripts/build_gallery_lasttoken.py
    uv run python scripts/build_gallery_lasttoken.py --max-samples 200   # smoke test
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

QWEN_MODEL_PATH = "/data/drdcad/sunggu/llm_models/Qwen3-Embedding-8B"
GALLERY_SENTENCES_PATH = "/data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/data/gallery_sentences.pt"
DEFAULT_OUTPUT_PATH = "/data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-finetune/data/gallery_embeddings_lasthidden.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None, help="Debug: only embed the first N sentences")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sentences = torch.load(GALLERY_SENTENCES_PATH, map_location="cpu")
    if args.max_samples is not None:
        sentences = sentences[: args.max_samples]
    print(f"Loaded {len(sentences)} gallery sentences from {GALLERY_SENTENCES_PATH}", flush=True)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model = AutoModel.from_pretrained(QWEN_MODEL_PATH, torch_dtype=torch.bfloat16)
    model.to(device).eval()

    all_embeds = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), args.batch_size), desc="Encoding gallery"):
            batch = sentences[i : i + args.batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            embeds = out.last_hidden_state[:, -1].float()  # last-token pooling (needs left padding)
            embeds = F.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())

    gallery_embeds = torch.cat(all_embeds, dim=0)
    elapsed = time.time() - t0
    print(f"Encoded {gallery_embeds.shape[0]} sentences -> {tuple(gallery_embeds.shape)} in {elapsed/60:.1f} min", flush=True)

    # Order must match gallery_sentences.pt row-for-row -- retrieval_metric.py indexes into
    # the gallery by position (pred_gallery_idx_all = argmax over gallery, then gallery_texts[idx]
    # to render it), so a silent reorder here would corrupt every recall@k / matched-phrase
    # number downstream without ever raising an error. Nothing in this script sorts, shuffles,
    # or otherwise reindexes sentences (batches are sentences[i:i+batch_size] in a plain
    # range() loop, HF's tokenizer(list, padding=True) does not reorder its input, and
    # torch.cat concatenates in append order) -- this assert makes that guarantee checkable
    # instead of just asserted in a comment.
    assert gallery_embeds.shape[0] == len(sentences), (
        f"row count mismatch: encoded {gallery_embeds.shape[0]} embeddings for "
        f"{len(sentences)} sentences -- order/count guarantee broken, do not save"
    )
    print(f"Order check: first sentence = {sentences[0]!r}", flush=True)
    print(f"Order check: last sentence  = {sentences[-1]!r}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write via a PID-suffixed temp file + os.replace: this file is what RetrievalMetric
    # loads at eval time, and this job runs for tens of minutes, so a half-written file must
    # never be visible. PID-suffixed (not a fixed ".tmp" name) because srun on this cluster has
    # been observed to launch 2 duplicate tasks for a single job step -- a shared fixed tmp name
    # let one task's rename race the other's (caught by smoke-testing this script before
    # submitting the full run: FileNotFoundError on the second task's rename).
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.pt")
    torch.save(gallery_embeds, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
