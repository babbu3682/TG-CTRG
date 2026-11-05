#!/usr/bin/env python3
"""병합 결과를 **문서 단위 read-back 으로 검증하고 기운다**.

왜 필요한가
-----------
`label_guided_merge.py --mode repair` 는 **문장 단위** 라벨로 문장을 고른다. 그런데 채점기는
**문서 전체**를 RadBERT 에 넣는다. 둘이 어긋나서, 800_x5b4 기준 투표 양성 7,447 개 중
168 개가 실현되지 않고 129 개가 투표에 없이 나타난다(일치율 98.94%). 그 작은 어긋남이
CRG+F1 을 **0.0082** 먹는다 (투표 목표 1.1816 -> 실현 1.1734).

이 스크립트는 그 간극만 닫는다: 병합본을 문서 단위로 다시 읽어 (a) 투표에 있는데 안 읽힌
라벨에 기증 문장을 더 붙이고 (b) 투표에 없는데 읽힌 라벨을 만드는 문장을 (안전할 때만) 뺀다.

투표·군집·문장라벨 수학은 `label_guided_merge` 에서 **import 한다.** 복제하면 병합 경로와
조용히 갈라진다.

안전장치
--------
- `--check_against <tag>`: 내 문서 단위 분류가 채점기의 `inferred_labels.csv` 를 재현하는지
  먼저 확인한다. 재현하지 못하면 피드백 신호가 채점기가 보는 것과 다르므로 **즉시 중단**한다.
- `--max_add`: exam 당 추가 문장 수 상한. 무한정 붙이면 BLEU 가 무너진다.
- `--drop_extra`: 기본 꺼짐. 문장 제거는 투표 라벨을 하나도 걸치지 않을 때만 한다.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_guided_merge import (  # noqa: E402
    LABEL_COLS, sections, cluster_block, load_classifier, classify,
    DEFAULT_RADBERT_CKPT, DEFAULT_RADBERT_DIR,
)

import re


def skey(s: str) -> str:
    """문장의 해시 가능한 정규화 키.

    label_guided_merge.nrm 은 Jaccard 용 **토큰 집합**을 돌려주므로 set/dict 키로 쓸 수 없다.
    풀은 이미 near-duplicate 군집을 거친 대표 문장들이라, "이미 들어 있는가" 판정에는
    정확 일치 키로 충분하다."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def render(f_sents, i_sents):
    out = "Findings: " + " ".join(f_sents)
    if i_sents:
        out += " Impression: " + " ".join(i_sents)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="label_guided_merge 산출 CSV (출발점)")
    ap.add_argument("--draws", required=True, nargs="+", help="기증 문장 풀 + 투표에 쓸 draw CSV 들")
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--max_add", type=int, default=3)
    ap.add_argument("--drop_extra", action="store_true")
    ap.add_argument("--ratio", type=float, default=0.80)
    ap.add_argument("--jaccard", type=float, default=0.60)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--radbert_model_path", default=DEFAULT_RADBERT_CKPT)
    ap.add_argument("--radbert_dir", default=DEFAULT_RADBERT_DIR)
    ap.add_argument("--label_cache", default="")
    ap.add_argument("--check_against", default="",
                    help="채점기 재현 확인용 tag (results/<dir>/<tag>/vlm3d_eval/inferred_labels.csv)")
    a = ap.parse_args()

    merged = pd.read_csv(a.merged).set_index("SampleName")
    dfs = [pd.read_csv(p).set_index("SampleName") for p in a.draws]
    names = list(merged.index)
    for p, d in zip(a.draws, dfs):
        miss = [n for n in names if n not in d.index]
        if miss:
            sys.exit(f"draw 커버리지 불일치 {p}: {len(miss)} missing (예: {miss[:3]})")

    # ---- 투표 (병합과 동일하게 채점기 라벨로) ----
    votes = []
    for p in a.draws:
        lp = Path(p).with_suffix("") / "vlm3d_eval" / "inferred_labels.csv"
        if not lp.exists():
            sys.exit(f"라벨 파일 없음: {lp}")
        votes.append(pd.read_csv(lp).set_index("AccessionNo")[LABEL_COLS])
    V = {n: (sum(v.loc[n].values.astype(int) for v in votes) >= a.min_support) for n in names}
    print(f"[vote] min_support {a.min_support}/{len(a.draws)}  "
          f"평균 {np.mean([V[n].sum() for n in names]):.2f} labels/exam", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)
    tok, model = load_classifier(a.radbert_dir, a.radbert_model_path, dev)

    # ---- 안전장치: 내 문서 단위 분류 == 채점기? ----
    if a.check_against:
        ref_p = Path(a.merged).with_suffix("").parent / a.check_against / "vlm3d_eval" / "inferred_labels.csv"
        if not ref_p.exists():
            sys.exit(f"--check_against 대상이 없다: {ref_p}")
        ref = pd.read_csv(ref_p).set_index("AccessionNo")[LABEL_COLS].loc[names].values.astype(int)
        mine = (classify([merged.loc[n, "Pred_report"] for n in names],
                         tok, model, dev, a.batch_size) >= a.threshold).astype(int)
        agree = (mine == ref).mean()
        print(f"[check] 문서 단위 재현 일치율 {agree*100:.4f}%  "
              f"(불일치 셀 {int((mine != ref).sum())})", flush=True)
        if agree < 0.999:
            sys.exit("[check] 채점기 라벨을 재현하지 못한다 -- 피드백 신호가 다르므로 중단한다")

    # ---- 문장 풀 (병합과 같은 군집) ----
    per, flat = {}, []
    for n in names:
        blocks = [sections(d.loc[n, "Pred_report"]) for d in dfs]
        f = cluster_block([b[0] for b in blocks], "Findings:", a.ratio, a.jaccard)
        i = cluster_block([b[1] for b in blocks], "Impression:", a.ratio, a.jaccard)
        per[n] = (f, i)
        for c in f + i:
            c["idx"] = len(flat)
            flat.append(c["rep"])
    print(f"[pool] {len(flat)} 문장 ({len(flat)/len(names):.1f}/exam)", flush=True)

    cache = Path(a.label_cache) if a.label_cache else None
    prob = None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["sents"]) == flat:
            prob = z["prob"]; print(f"[label] 캐시 재사용 {cache}", flush=True)
        else:
            print("[label] 캐시 불일치 -- 다시 분류한다", flush=True)
    if prob is None:
        prob = classify(flat, tok, model, dev, a.batch_size)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, prob=prob, sents=np.array(flat, dtype=object))
    lab = prob >= a.threshold

    # ---- 현재 문서를 문장 리스트로 ----
    cur = {}
    for n in names:
        f, i = sections(merged.loc[n, "Pred_report"])
        from label_guided_merge import sents as _sents
        cur[n] = [_sents(f, "Findings:"), _sents(i, "Impression:")]

    for it in range(1, a.iters + 1):
        texts = [render(*cur[n]) for n in names]
        A = (classify(texts, tok, model, dev, a.batch_size) >= a.threshold)
        miss = sum(int((V[n] & ~A[k]).sum()) for k, n in enumerate(names))
        extra = sum(int((~V[n] & A[k]).sum()) for k, n in enumerate(names))
        print(f"[iter {it}] 미실현 {miss}  초과 {extra}", flush=True)
        if miss == 0 and (extra == 0 or not a.drop_extra):
            print("[iter] 더 고칠 것이 없다"); break

        n_add = n_drop = 0
        for k, n in enumerate(names):
            v, act = V[n], A[k]
            have = {skey(s) for s in cur[n][0] + cur[n][1]}
            pool = per[n][0] + per[n][1]
            nF = len(per[n][0])
            added_here = 0
            for li in np.where(v & ~act)[0]:
                if added_here >= a.max_add:
                    break                      # exam 당 상한 -- 무한정 붙이면 BLEU 가 무너진다
                cand = [(j, c) for j, c in enumerate(pool)
                        if skey(c["rep"]) not in have and lab[c["idx"]][li]
                        and not (lab[c["idx"]] & ~v).any()]
                if not cand:
                    cand = [(j, c) for j, c in enumerate(pool)
                            if skey(c["rep"]) not in have and lab[c["idx"]][li]]
                if not cand:
                    continue
                j, c = min(cand, key=lambda t: (int(lab[t[1]["idx"]].sum()), len(t[1]["rep"])))
                cur[n][0 if j < nF else 1].append(c["rep"])
                have.add(skey(c["rep"])); n_add += 1; added_here += 1
            if a.drop_extra:
                for sec in (0, 1):
                    keep = []
                    for s in cur[n][sec]:
                        hit = [c for c in pool if skey(c["rep"]) == skey(s)]
                        L = lab[hit[0]["idx"]] if hit else np.zeros(len(LABEL_COLS), bool)
                        if L.any() and not (L & v).any():
                            n_drop += 1; continue
                        keep.append(s)
                    cur[n][sec] = keep
        print(f"[iter {it}] 추가 {n_add}  제거 {n_drop}", flush=True)
        if n_add == 0 and n_drop == 0:
            print("[iter] 변화 없음 -- 중단"); break

    out = merged.copy()
    out["Pred_report"] = [render(*cur[n]) for n in names]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(a.out, index=False)
    b0 = np.mean([len(merged.loc[n, "Pred_report"]) for n in names])
    b1 = np.mean([len(t) for t in out["Pred_report"]])
    print(f"[done] {a.out}  길이 {b0:.0f} -> {b1:.0f} 자 ({b1/b0:.3f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
