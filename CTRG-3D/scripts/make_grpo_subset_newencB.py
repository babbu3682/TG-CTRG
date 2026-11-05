#!/usr/bin/env python3
"""Freeze the RL prompt subset for GRPO-on-newencB. CPU only, seconds.

    python scripts/make_grpo_subset_newencB.py

This subset is the definition domain of three things at once, which is why it is
frozen to a file before anything else runs:

  1. the BLEU paired floor        (the starting policy's per-volume BLEU)
  2. the macro-F1 corpus context  (the starting policy's predicted labels)
  3. the prompts GRPO ever sees

If any of the three were drawn separately they would drift apart and the reward
would score rollouts against a corpus that does not contain them.

SIZE: 12,000 of the ~23,850 cached train volumes (50%). At 8 prompts/step that is
1,500 steps = still exactly 1 EPOCH -- double round 2's update budget without a
second pass over the same prompts. Round 2 ended 1 epoch still improving, but with
entropy 0.687 -> 0.392 and frac_reward_zero_std leaving 0 after step 601, so a 2nd
epoch risks group-variance collapse; fresh prompts do not. Reference generation
costs ~4.7 h at the measured 1.42 s/volume.

DEADLINE (2026-08-30): 16,000 was the first choice, but reference generation is on
the critical path before RL can start at all. 12,000 keeps 1-epoch semantics and
buys back 1.6 h. The run itself does not have to finish -- save_steps=100 means the
best checkpoint at deadline is what ships.

SAMPLING: uniform, from a DEDICATED seed (not the global RNG). Deliberately not
oversampling rare labels: the corpus prevalence sets r = (X-A)/(2A) and SIGMA_F1,
and skewing it would make both diverge from the valid split the run is scored on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
SEED = 20260826          # dedicated; never the global RNG, and not round 2's
N_SUBSET = 12000
CACHE_INDEX = _REPO / "data/vision_cache/newencB_ep15_ckpt7455/train_index.json"
ORG_LABELS = "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/train_predicted_labels.csv"
OUT_DIR = _REPO / "data/grpo_ref_newencB"


def main() -> None:
    import sys
    sys.path.insert(0, str(_REPO / "scripts"))
    from reward_challenge_metrics import LABEL_COLS, N_LABELS, load_organizer_labels

    index = json.loads(CACHE_INDEX.read_text())
    labels = load_organizer_labels(ORG_LABELS)

    # Every RL volume needs BOTH a cached visual row and an organiser label row;
    # a volume missing either would raise mid-training instead of here.
    eligible = sorted(n for n in index if n in labels)
    missing = sorted(set(index) - set(labels))
    if missing:
        raise RuntimeError(
            f"{len(missing)} cached train volumes have no organiser label row, "
            f"e.g. {missing[:3]}. The reward cannot score them.")
    if len(eligible) < N_SUBSET:
        raise RuntimeError(f"only {len(eligible)} eligible volumes, need {N_SUBSET}")

    rng = np.random.default_rng(SEED)
    picked = sorted(rng.choice(eligible, size=N_SUBSET, replace=False).tolist())

    G = np.array([labels[n] for n in picked])
    X, A = N_LABELS * len(picked), int(G.sum())
    r = (X - A) / (2 * A)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "subset12000.json").write_text(json.dumps(picked, indent=0))
    meta = {
        "seed": SEED,
        "n": len(picked),
        "n_eligible": len(eligible),
        "cache_index": str(CACHE_INDEX),
        "organiser_labels": ORG_LABELS,
        # r and U are GT-only constants; the reward needs them and they must come
        # from THIS subset, not from the valid split.
        "X": X, "A": A, "r": r, "U": (X - A) / 2,
        "breakeven_p": 1.0 / (2 * r + 1),
        "prevalence": {LABEL_COLS[j]: float(G[:, j].mean()) for j in range(N_LABELS)},
    }
    (OUT_DIR / "subset12000_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[subset] {len(picked)} volumes  seed={SEED}")
    print(f"[crg]    X={X} A={A} r={r:.4f} U={meta['U']:.1f} "
          f"breakeven p={100 * meta['breakeven_p']:.2f}%")
    print(f"[out]    {OUT_DIR}/subset12000.json")


if __name__ == "__main__":
    main()
