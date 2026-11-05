"""Label-guided merge + read-back verify loop — the ensemble's whole method.

Port of PETRG-3D ``scripts/label_guided_merge.py`` (mode ``repair``) and
``scripts/repair_verify_loop.py``, restructured to take draws in memory instead
of reading a CSV per draw and an ``inferred_labels.csv`` per draw out of a
scoring directory. The maths is unchanged; only the plumbing moved, because a
test run has no scorer to read labels back from.

THE IDEA, IN ONE PARAGRAPH
--------------------------
Ten draws of the same exam say the same findings in different words, so voting
on SENTENCES throws away true findings to paraphrase (measured upstream:
recall 0.588 -> 0.495). So the vote happens in LABEL space — the 18 labels
RadBERT reads out of each whole draft — and the sentences are then filtered and
repaired to realise that vote. The skeleton draw supplies the prose; sentences
claiming only un-voted findings are dropped; voted labels that no surviving
sentence carries are filled from the other draws' sentences.

Then the verify loop closes the remaining gap: the merge filters with
SENTENCE-level labels, but the scorer reads WHOLE DOCUMENTS. Upstream that
mismatch left ~1.4% of voted positives unrealised and cost 0.0082 of CRG+F1.
The loop re-labels the merged document and patches the difference.

NOTHING HERE READS GROUND TRUTH. The vote's input is RadBERT run on generated
text; the only things carried over from the valid split are three scalars —
``min_support`` (3), which draw is the skeleton (index 6), and the add budgets.
That is what makes a recipe fitted on valid applicable to the test set.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable, Sequence

import numpy as np

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
N_LAB = len(LABEL_COLS)

SPLIT = re.compile(r'(?<=[.!?])\s+')


# --------------------------------------------------------------------------
# text utilities — verbatim from label_guided_merge.py
# --------------------------------------------------------------------------

def sections(txt):
    t = (txt or "").strip()
    i = t.find("Impression:")
    return (t, "") if i < 0 else (t[:i].rstrip(), t[i:].strip())


def sents(block, prefix):
    b = block.strip()
    if prefix and b.startswith(prefix):
        b = b[len(prefix):].strip()
    return [s.strip() for s in SPLIT.split(b) if s.strip()]


def nrm(s):
    return set(re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split())


def same(a, b, na, nb, ratio, jac):
    if na and nb and len(na & nb) / len(na | nb) >= jac:
        return True
    return SequenceMatcher(None, a, b).ratio() >= ratio


def cluster_block(blocks, prefix, ratio, jac):
    """draw-wise blocks -> [{'rep','pos','draws'}] near-duplicate clusters."""
    cl = []
    for di, blk in enumerate(blocks):
        ss = sents(blk, prefix)
        n = max(len(ss), 1)
        for si, s in enumerate(ss):
            ns = nrm(s)
            hit = None
            for c in cl:
                if same(s, c['rep'], ns, c['norm'], ratio, jac):
                    hit = c
                    break
            if hit is None:
                cl.append({'rep': s, 'norm': ns, 'draws': {di}, 'pos': [si / n]})
            else:
                hit['draws'].add(di)
                hit['pos'].append(si / n)
    cl.sort(key=lambda c: sum(c['pos']) / len(c['pos']))
    return cl


def skey(s: str) -> str:
    """Hashable normalised key for a sentence.

    ``nrm`` returns a token SET for Jaccard and cannot be a dict key. The pool
    is already de-duplicated into cluster representatives, so exact-match keys
    are enough for "is this already in the document?".
    """
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def render(f_sents, i_sents):
    out = "Findings: " + " ".join(f_sents)
    if i_sents:
        out += " Impression: " + " ".join(i_sents)
    return out


# --------------------------------------------------------------------------
# stage D — label-guided merge (mode "repair")
# --------------------------------------------------------------------------

def build_pool(names: Sequence[str], draws: Sequence[dict],
               ratio: float = 0.80, jaccard: float = 0.60):
    """Cluster every draw's sentences per exam; return (per_exam, flat_sentences)."""
    per, flat = {}, []
    for n in names:
        blocks = [sections(d[n]) for d in draws]
        f = cluster_block([b[0] for b in blocks], "Findings:", ratio, jaccard)
        i = cluster_block([b[1] for b in blocks], "Impression:", ratio, jaccard)
        per[n] = (f, i)
        for c in f + i:
            c['idx'] = len(flat)
            flat.append(c['rep'])
    return per, flat


def vote(names: Sequence[str], doc_labels: np.ndarray, min_support: int):
    """doc_labels: (n_draws, N, 18) binary -> (V, VC) per exam.

    ``V`` is the boolean "at least ``min_support`` draws claimed it" vector;
    ``VC`` is the raw count, which orders the fill when the add budget binds.
    """
    counts = doc_labels.astype(int).sum(axis=0)          # (N, 18)
    V = {n: counts[k] >= min_support for k, n in enumerate(names)}
    VC = {n: counts[k] for k, n in enumerate(names)}
    return V, VC


def merge(names: Sequence[str], draws: Sequence[dict], doc_labels: np.ndarray,
          sent_lab: np.ndarray, per: dict, *, min_support: int, backbone: int,
          max_add: int = 0, max_add_impression: int = -999) -> dict:
    """Return {name: merged report text}.

    ``max_add``            per-exam added-sentence cap, 0 = unlimited.
    ``max_add_impression`` Impression-only cap; -999 = share ``max_add``,
                           0 = unlimited, -1 = add nothing. Impression averages
                           2.9 GT sentences, so one extra there is +34% where
                           the same sentence in Findings is +6%; sharing one
                           budget lets Impression overshoot and costs BLEU.
    """
    V, VC = vote(names, doc_labels, min_support)

    def repair(cl, v, vc, budget):
        base_c = [c for c in cl if backbone in c['draws']]
        out, covered = [], np.zeros(N_LAB, bool)
        for c in base_c:
            L = sent_lab[c['idx']]
            if L.any() and not (L & v).any():
                continue                      # claims only un-voted findings
            out.append(c)
            covered |= (L & v)
        todo = np.where(v & ~covered)[0]
        if vc is not None:
            todo = todo[np.argsort(-vc[todo], kind="stable")]
        for li in todo:
            if budget is not None and budget[0] <= 0:
                break
            cand = [c for c in cl if c not in out and sent_lab[c['idx']][li]
                    and not (sent_lab[c['idx']] & ~v).any()]
            if not cand:
                cand = [c for c in cl if c not in out and sent_lab[c['idx']][li]]
            if not cand:
                continue
            c = min(cand, key=lambda c: (int(sent_lab[c['idx']].sum()), len(c['rep'])))
            out.append(c)
            covered |= (sent_lab[c['idx']] & v)
            if budget is not None:
                budget[0] -= 1
        out.sort(key=lambda c: sum(c['pos']) / len(c['pos']))
        return [c['rep'] for c in out]

    merged = {}
    for n in names:
        v = V[n]
        f, i = per[n]
        bud = [max_add] if max_add else None
        fs = repair(f, v, VC[n], bud)
        bud_i = bud
        if max_add_impression != -999:
            bud_i = [max_add_impression] if max_add_impression else None
        is_ = repair(i, v, VC[n], bud_i)
        if not fs:
            fs = [c['rep'] for c in f if backbone in c['draws']] or \
                 [c['rep'] for c in f]
        txt = "Findings: " + " ".join(fs)
        if is_:
            txt += "\nImpression: " + " ".join(is_)
        merged[n] = txt
    return merged


# --------------------------------------------------------------------------
# stage E — read-back verify loop
# --------------------------------------------------------------------------

def verify_loop(names: Sequence[str], merged: dict, doc_labels: np.ndarray,
                sent_lab: np.ndarray, per: dict,
                label_documents: Callable[[list], np.ndarray], *,
                min_support: int, iters: int = 2, max_add: int = 2,
                drop_extra: bool = False, log=print) -> dict:
    """Re-label the merged documents and patch what the vote asked for but the
    document-level labeller does not read back.

    ``label_documents`` must be the SAME whole-document labeller the scorer
    uses — a sentence-level proxy here would optimise a different signal.
    """
    V, _ = vote(names, doc_labels, min_support)

    cur = {}
    for n in names:
        f, i = sections(merged[n])
        cur[n] = [sents(f, "Findings:"), sents(i, "Impression:")]

    for it in range(1, iters + 1):
        texts = [render(*cur[n]) for n in names]
        A = label_documents(texts).astype(bool)
        miss = sum(int((V[n] & ~A[k]).sum()) for k, n in enumerate(names))
        extra = sum(int((~V[n] & A[k]).sum()) for k, n in enumerate(names))
        log(f"[iter {it}] unrealised {miss}  excess {extra}")
        if miss == 0 and (extra == 0 or not drop_extra):
            log("[iter] nothing left to fix")
            break

        n_add = n_drop = 0
        for k, n in enumerate(names):
            v, act = V[n], A[k]
            have = {skey(s) for s in cur[n][0] + cur[n][1]}
            pool = per[n][0] + per[n][1]
            nF = len(per[n][0])
            added_here = 0
            for li in np.where(v & ~act)[0]:
                if added_here >= max_add:
                    break
                cand = [(j, c) for j, c in enumerate(pool)
                        if skey(c["rep"]) not in have and sent_lab[c["idx"]][li]
                        and not (sent_lab[c["idx"]] & ~v).any()]
                if not cand:
                    cand = [(j, c) for j, c in enumerate(pool)
                            if skey(c["rep"]) not in have and sent_lab[c["idx"]][li]]
                if not cand:
                    continue
                j, c = min(cand, key=lambda t: (int(sent_lab[t[1]["idx"]].sum()),
                                                len(t[1]["rep"])))
                cur[n][0 if j < nF else 1].append(c["rep"])
                have.add(skey(c["rep"]))
                n_add += 1
                added_here += 1
            if drop_extra:
                for sec in (0, 1):
                    keep = []
                    for s in cur[n][sec]:
                        hit = [c for c in pool if skey(c["rep"]) == skey(s)]
                        L = sent_lab[hit[0]["idx"]] if hit else np.zeros(N_LAB, bool)
                        if L.any() and not (L & v).any():
                            n_drop += 1
                            continue
                        keep.append(s)
                    cur[n][sec] = keep
        log(f"[iter {it}] added {n_add}  dropped {n_drop}")
        if n_add == 0 and n_drop == 0:
            log("[iter] no change — stopping")
            break

    return {n: render(*cur[n]) for n in names}
