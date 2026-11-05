#!/usr/bin/env python
"""라벨 유도 병합 -- 라벨 공간의 VOTE>=k 결과를 텍스트로 실현한다.

배경. ckpt-600 의 4 draw 를 **라벨 공간**에서 조합하면

    VOTE>=1 (union)   CRG 0.5587  macroF1 0.5799   (P 0.476 / R 0.757)
    VOTE>=2 of 4      CRG 0.5521  macroF1 0.6002   (P 0.563 / R 0.658)  <- 두 목표 동시 충족

인데, 같은 규칙을 **문장 유사도**로 흉내낸 scripts/consensus_merge.py 는
CRG 0.4849 / F1 0.5131 로 크게 못 미쳤다. 원인은 측정으로 확인됐다: 서로 다른
draw 는 같은 소견을 다르게 표현하므로(paraphrase) 문장 군집이 갈라지고,
"2개 draw 가 말한 문장" 조건이 참 소견까지 버린다 -- recall 0.588 -> 0.495.

그래서 투표는 **라벨 공간에서** 하고(= 채점기가 실제로 읽어낸 라벨,
vlm3d_eval/inferred_labels.csv), 문장은 그 투표 결과를 만족하도록 **거른다**:

    V = {>= min_support 개 draw 가 주장한 라벨}
    후보 = 모든 draw 의 문장 (near-duplicate 군집)
    keep(문장) = 라벨이 없거나(보일러플레이트/음성 -- BLEU 를 지탱한다)
                 strict: 라벨 집합이 V 에 포함될 때
                 loose : 라벨 집합이 V 와 교집합이 있을 때

문장 라벨은 scripts/label_report_sentences.py 의 분류기를 그대로 import 해서 쓴다
(채점기와 같은 RadBertClassifier 체크포인트·threshold). 다른 분류기를 쓰면
지표의 대리(proxy)가 되어 버린다.

주의 -- 문장 단위 입력은 분류기의 분포 밖이다(리포트 전체로 학습됐다). 이 스크립트는
그 라벨을 **정답으로 쓰지 않고 후보를 거르는 데만** 쓰며, 최종 판정은 언제나
src/evaluate_reportgen.py 의 full-report 채점이다.
"""
import argparse, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_report_sentences import (  # noqa: E402
    LABEL_COLS, DEFAULT_RADBERT_CKPT, DEFAULT_RADBERT_DIR,
    load_classifier, classify,
)

SPLIT = re.compile(r'(?<=[.!?])\s+')


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
    """draw 별 블록 -> [{'rep','pos','draws'}] (near-duplicate 군집, 중복 제거만 한다)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", required=True, nargs="+",
                    help="draw CSV 들. 각각 옆에 <dir>/vlm3d_eval/inferred_labels.csv 가 있어야 한다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--mode", choices=["strict", "loose", "repair"], default="repair")
    ap.add_argument("--backbone", type=int, default=0, help="repair 모드의 뼈대 draw 인덱스")
    ap.add_argument("--max_add", type=int, default=0,
                    help="repair: exam 당 추가 문장 상한 (0=무제한=기존 동작). 붙일수록 라벨은 오르고\n                         BLEU 는 내려간다 -- 합의 스윙 요인이 BLEU 라서 이 축이 k 보다 직접적이다.\n                         상한이 걸리면 **투표 수가 많은 라벨부터** 채운다.")
    ap.add_argument("--drop_redundant", action="store_true",
                    help="repair: 자기 투표라벨이 이미 다른 문장으로 전부 덮이는 문장을 뺀다.\n                         라벨 집합을 바꾸지 않고 길이만 줄인다 -- 합의 제약이 길이라서 이 방향이\n                         라벨/BLEU 1:1 상쇄를 벗어날 수 있다. 라벨 없는 보일러플레이트는 BLEU 를\n                         지탱하므로 건드리지 않는다.")
    ap.add_argument("--max_add_impression", type=int, default=-999,
                    help="Impression 섹션 전용 추가 상한. 미지정(-999)이면 --max_add 를 두 섹션이 공유,\n                         0=무제한, -1=추가 없음 (--max_add 와 같은 규약).\n                         Impression 은 GT 평균 2.9 문장뿐이라 1 문장만 더해도 +34%% 인데 Findings 는\n                         +6%% 다 -- 같은 예산을 공유하면 Impression 쪽 과다가 BLEU 를 깎는다.")
    ap.add_argument("--backbone_per_exam", action="store_true",
                    help="exam 마다 투표 결과에 **가장 가까운** draw 를 뼈대로 쓴다(해밍 거리 최소,\n                         동률이면 --backbone 을 선호). 수리량이 줄어 더 짧고 일관된 텍스트가 되고,\n                         GT 를 보지 않으므로 test 에도 그대로 적용된다.")
    ap.add_argument("--bb_margin", type=int, default=0,
                    help="--backbone_per_exam 의 관성. 다른 draw 가 기본 뼈대보다 해밍 거리로 이 값을\n                         **초과해서** 가까울 때만 바꾼다(0=동률이면 기본 유지). 기본 뼈대는 BLEU 가\n                         가장 좋은 draw 이므로, 관성을 주면 BLEU 를 더 지키면서 명백히 유리한 exam 만\n                         갈아탄다.")
    ap.add_argument("--max_drop_redundant", type=int, default=0,
                    help="--drop_redundant 의 exam 당 상한 (0=무제한). 실측상 중복 후보가 문장의 35%%,\n                         문자의 38%% 라 무제한이면 길이비가 1.067 -> 0.66 으로 폭락해 brevity penalty\n                         로 BLEU 가 무너진다. 길이비를 1.02~1.03 로 맞추려면 1~2 개면 된다.")
    ap.add_argument("--ratio", type=float, default=0.80)
    ap.add_argument("--jaccard", type=float, default=0.60)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--radbert_model_path", default=DEFAULT_RADBERT_CKPT)
    ap.add_argument("--radbert_dir", default=DEFAULT_RADBERT_DIR)
    ap.add_argument("--label_cache", default="", help="문장 라벨 캐시 .npz (풀이 같으면 재사용)")
    a = ap.parse_args()

    if len(a.draws) < 2:
        sys.exit("draw 가 2개 미만이다")

    dfs = [pd.read_csv(p).set_index("SampleName") for p in a.draws]
    base = dfs[0]
    names = list(base.index)
    for d in dfs[1:]:
        miss = [n for n in names if n not in d.index]
        if miss:
            sys.exit(f"draw 커버리지 불일치: {len(miss)} missing (예: {miss[:3]})")

    # 투표는 채점기가 실제로 읽어낸 라벨로 한다.
    votes = []
    for p in a.draws:
        lp = Path(p).with_suffix("") / "vlm3d_eval" / "inferred_labels.csv"
        if not lp.exists():
            sys.exit(f"라벨 파일 없음: {lp} -- 먼저 그 draw 를 채점해야 한다")
        votes.append(pd.read_csv(lp).set_index("AccessionNo")[LABEL_COLS])
    V, VC = {}, {}
    for n in names:
        s = np.zeros(len(LABEL_COLS), dtype=int)
        for v in votes:
            s += v.loc[n].values.astype(int)
        V[n] = (s >= a.min_support)
        VC[n] = s                      # 투표 수 -- --max_add 우선순위에 쓴다
    print(f"[vote] min_support {a.min_support}/{len(a.draws)}  "
          f"평균 {np.mean([V[n].sum() for n in names]):.2f} labels/exam", flush=True)

    # 군집 -> 전역 문장 리스트 (분류기를 꽉 찬 배치로 돌리기 위해)
    per, flat = {}, []
    for n in names:
        blocks = [sections(d.loc[n, "Pred_report"]) for d in dfs]
        f = cluster_block([b[0] for b in blocks], "Findings:", a.ratio, a.jaccard)
        i = cluster_block([b[1] for b in blocks], "Impression:", a.ratio, a.jaccard)
        per[n] = (f, i)
        for c in f + i:
            c['idx'] = len(flat)
            flat.append(c['rep'])
    print(f"[pool] {len(flat)} 문장 ({len(flat)/len(names):.1f}/exam)", flush=True)

    cache = Path(a.label_cache) if a.label_cache else None
    prob = None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["sents"]) == flat:          # 풀이 바이트 단위로 같을 때만 재사용
            prob = z["prob"]
            print(f"[label] 캐시 재사용 {cache}", flush=True)
        else:
            print(f"[label] 캐시 불일치 -- 다시 분류한다", flush=True)
    if prob is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[device] {dev}", flush=True)
        tok, model = load_classifier(a.radbert_dir, a.radbert_model_path, dev)
        prob = classify(flat, tok, model, dev, a.batch_size)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, prob=prob, sents=np.array(flat, dtype=object))
    lab = prob >= a.threshold
    print(f"[label] 라벨 없는 문장 {(~lab.any(1)).mean()*100:.1f}%", flush=True)

    def keep(c, v):
        L = lab[c['idx']]
        if not L.any():
            return True
        return (not (L & ~v).any()) if a.mode == "strict" else bool((L & v).any())

    bb = a.backbone

    def repair(cl, v, vc=None, budget=None, bb_i=None):
        """뼈대 draw 의 문장에서 시작해 (1) 투표에 없는 소견만 말하는 문장을 빼고
        (2) 투표에 있는데 빠진 라벨을 다른 draw 문장으로 채운다. 길이가 뼈대 근처로 남는다."""
        base_c = [c for c in cl if (bb if bb_i is None else bb_i) in c['draws']]
        out, covered, dropped = [], np.zeros(len(LABEL_COLS), bool), 0
        for c in base_c:
            L = lab[c['idx']]
            if L.any() and not (L & v).any():
                dropped += 1
                continue                        # 투표에 없는 소견만 주장 -> 뺀다
            out.append(c)
            covered |= (L & v)
        added = 0
        todo = np.where(v & ~covered)[0]
        if vc is not None:
            todo = todo[np.argsort(-vc[todo], kind="stable")]   # 투표 수 큰 라벨부터
        for li in todo:
            if budget is not None and budget[0] <= 0:
                break                      # exam 전체(Findings+Impression) 공유 예산
            cand = [c for c in cl if c not in out and lab[c['idx']][li]
                    and not (lab[c['idx']] & ~v).any()]
            if not cand:
                cand = [c for c in cl if c not in out and lab[c['idx']][li]]
            if not cand:
                continue
            c = min(cand, key=lambda c: (int(lab[c['idx']].sum()), len(c['rep'])))
            out.append(c); covered |= (lab[c['idx']] & v); added += 1
            if budget is not None:
                budget[0] -= 1
        if a.drop_redundant:
            changed, n_red = True, 0
            while changed and (not a.max_drop_redundant or n_red < a.max_drop_redundant):
                changed = False
                for c in sorted(out, key=lambda z: -len(z['rep'])):   # 긴 것부터
                    L = lab[c['idx']] & v
                    if not L.any():
                        continue                    # 보일러플레이트는 남긴다
                    others = np.zeros(len(LABEL_COLS), bool)
                    for d in out:
                        if d is not c:
                            others |= (lab[d['idx']] & v)
                    if not (L & ~others).any():      # 이 문장이 유일하게 가진 투표라벨이 없다
                        out.remove(c); dropped += 1; n_red += 1; changed = True
                        break
        out.sort(key=lambda c: sum(c['pos']) / len(c['pos']))
        return [c['rep'] for c in out], len(out), len(base_c), dropped, added

    rows, kept, tot, n_drop, n_add = [], 0, 0, 0, 0
    for n in names:
        v = V[n]
        f, i = per[n]
        if a.mode == "repair":
            bud = [a.max_add] if a.max_add else None
            bbn = None
            if a.backbone_per_exam:
                dist = [int((votes[j].loc[n].values.astype(bool) != v).sum())
                        for j in range(len(votes))]
                best = min(dist)
                # 기본 뼈대가 best 보다 bb_margin 이하로만 나쁘면 유지 (BLEU 관성)
                bbn = bb if dist[bb] - best <= a.bb_margin else int(np.argmin(dist))
            fs, k1, t1, d1, a1 = repair(f, v, VC[n], bud, bbn)
            bud_i = bud
            if a.max_add_impression != -999:
                bud_i = [a.max_add_impression] if a.max_add_impression else None
            is_, k2, t2, d2, a2 = repair(i, v, VC[n], bud_i, bbn)
            kept += k1 + k2; tot += t1 + t2; n_drop += d1 + d2; n_add += a1 + a2
        else:
            fs = [c['rep'] for c in f if keep(c, v)]
            is_ = [c['rep'] for c in i if keep(c, v)]
            kept += len(fs) + len(is_)
            tot += len(f) + len(i)
        if not fs:
            _b = bb if not a.backbone_per_exam or bbn is None else bbn
            fs = [c['rep'] for c in f if _b in c['draws']] or [c['rep'] for c in f]
        txt = "Findings: " + " ".join(fs)
        if is_:
            txt += "\nImpression: " + " ".join(is_)
        rows.append({"SampleName": n,
                     "Question": base.loc[n, "Question"],
                     "GT_report": base.loc[n, "GT_report"],
                     "Pred_report": txt})
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    L0 = np.mean([len(str(base.loc[n, "Pred_report"])) for n in names])
    L1 = out["Pred_report"].str.len().mean()
    if a.mode == "repair":
        print(f"[repair] 뼈대에서 뺀 문장 {n_drop} ({n_drop/len(names):.2f}/exam), "
              f"채운 문장 {n_add} ({n_add/len(names):.2f}/exam)", flush=True)
    print(f"[{a.mode} >={a.min_support}] {len(out)} rows  문장 {kept}/{tot} 유지 "
          f"({kept/max(tot,1)*100:.1f}%)  길이 {L0:.0f} -> {L1:.0f} ({L1/L0:.2f}x)  -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
