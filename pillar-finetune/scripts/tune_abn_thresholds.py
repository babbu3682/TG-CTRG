"""abn 18개 라벨의 클래스별 최적 임계값을 찾는다 (전역 단일 임계값과 비교).

`scripts/build_cls_report_templates.py`와 완전히 같은 확률을 쓴다 -- 캐싱된 pooled
feature(abn_probe_cache/<tag>/{train,dev}.pt)에 probe_head.pt만 태워 sigmoid를 계산하므로
인코더/GPU가 필요 없고 전체가 CPU 몇 초다.

측정하는 정책 (모두 dev에서 채점):
  1. global 0.5 / global 0.7            -- 현재 쓰는 값
  2. global best                        -- macro-F1을 최대화하는 "하나의" 임계값 (retrieval 0.4 대응)
  3. per-class (train fit)              -- 클래스별 F1 최대 임계값을 train에서 적합
  4. per-class (dev fit, 낙관 상한)     -- dev에서 적합해서 dev에서 채점 -> 순환, 상한값으로만
  5. per-class (dev 5-fold CV)          -- dev 안에서 4/5 적합 -> 1/5 채점, 일반화 추정치

경고: 클래스별 임계값은 희소 클래스에서 과적합한다 (docs/abn-report-templates.md의 split-half
붕괴 기록 참고). 그래서 (4)와 (5)를 같이 낸다 -- (4)와 (5)의 격차가 곧 과적합 폭이다.

출력: <output-dir>/thresholds.json  (정책별 임계값 + 클래스별 지표)
      표는 stdout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# pillar 패키지를 import하면 kornia -> flash_attn 경로가 컨테이너 밖에서 GLIBC 에러를 내므로
# 라벨 순서만 pillar/metrics/abn_metric.py 정의 그대로 복사해둔다 (바뀌면 같이 고칠 것).
LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]

DEFAULT_CACHE = "/data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-pretrain/data/abn_probe_cache/original_pillar_lz4_half"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE)
    p.add_argument("--head-file", default="probe_head.pt",
                   help="cache-dir 안의 head state_dict 파일 이름. probe 계열은 "
                        "probe_head.pt(Sequential 인덱스 0/3), full-finetune 체크포인트에서 "
                        "뽑은 head는 finetune_head.pt처럼 별도 이름을 쓴다.")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--ppv-target", type=float, default=0.90,
                   help="고신뢰 positive 문장용: 이 PPV를 만족하는 최소 임계값도 같이 보고")
    p.add_argument("--npv-target", type=float, default=0.98)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--boot", type=int, default=2000, help="paired bootstrap 반복수 (0이면 생략)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="logs/CTRATE/CTRATE_ABN_threshold_tuning")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 확률 계산 (build_cls_report_templates.py와 동일)
# --------------------------------------------------------------------------- #
def load_probs(cache, hidden, split, head_file):
    blob = torch.load(cache / f"{split}.pt", map_location="cpu", weights_only=False)
    X, y = blob["features"], blob["labels"]
    state = torch.load(cache / head_file, map_location="cpu", weights_only=False)
    head = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(0.0),
                         nn.Linear(hidden, y.shape[1]))
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        probs = torch.sigmoid(head(X)).numpy().astype(np.float64)
    return probs, y.numpy().astype(np.int64), list(blob["sample_names"])


def auroc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- #
# 임계값 스윕: 정렬 한 번으로 모든 후보 임계값의 혼동행렬을 구한다
# --------------------------------------------------------------------------- #
def sweep(y, s):
    """반환: thr[K], tp[K], fp[K], fn[K], tn[K]  (thr 이상을 positive로 예측)."""
    order = np.argsort(-s, kind="mergesort")   # 확률 내림차순
    ys, ss = y[order], s[order]
    tp_c = np.cumsum(ys)
    fp_c = np.cumsum(1 - ys)
    # 동점 처리: 같은 확률의 마지막 위치만 남긴다
    keep = np.r_[np.diff(ss) != 0, True]
    thr = ss[keep]
    tp, fp = tp_c[keep], fp_c[keep]
    n1, n0 = y.sum(), len(y) - y.sum()
    return thr, tp, fp, n1 - tp, n0 - fp


def stats_at(tp, fp, fn, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        npv = np.where(tn + fn > 0, tn / (tn + fn), np.nan)
    return prec, rec, f1, npv


def f1_at(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0, tp, fp, fn


def best_f1_thr(y, s):
    """F1을 최대화하는 임계값. 동률이면 임계값이 큰 쪽(=보수적)을 택한다."""
    thr, tp, fp, fn, tn = sweep(y, s)
    _, _, f1, _ = stats_at(tp, fp, fn, tn)
    k = int(np.argmax(f1))            # argmax는 첫 번째 = 가장 높은 임계값 (thr 내림차순)
    return float(thr[k]), float(f1[k])


def best_youden_thr(y, s):
    thr, tp, fp, fn, tn = sweep(y, s)
    tpr = tp / max(y.sum(), 1)
    fpr = fp / max(len(y) - y.sum(), 1)
    k = int(np.argmax(tpr - fpr))
    return float(thr[k]), float((tpr - fpr)[k])


def thr_for_ppv(y, s, target):
    """PPV >= target을 만족하는 가장 낮은 임계값 (= 커버리지 최대). 없으면 None."""
    thr, tp, fp, fn, tn = sweep(y, s)
    prec, _, _, _ = stats_at(tp, fp, fn, tn)
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return None, None, None
    k = int(ok[-1])                    # thr 내림차순이므로 마지막이 가장 낮은 임계값
    return float(thr[k]), float(prec[k]), int(tp[k] + fp[k])


def thr_for_npv(y, s, target):
    """NPV >= target을 만족하는 가장 높은 임계값 (= negative 커버리지 최대). 없으면 None."""
    thr, tp, fp, fn, tn = sweep(y, s)
    _, _, _, npv = stats_at(tp, fp, fn, tn)
    ok = np.where(npv >= target)[0]
    if len(ok) == 0:
        return None, None, None
    k = int(ok[0])                     # 가장 높은 임계값
    return float(thr[k]), float(npv[k]), int(tn[k] + fn[k])


# --------------------------------------------------------------------------- #
def macro_f1(y, probs, thrs):
    return float(np.mean([f1_at(y[:, i], probs[:, i], thrs[i])[0] for i in range(y.shape[1])]))


def global_best_thr(y, probs, grid):
    """macro-F1을 최대화하는 단일 전역 임계값 (retrieval의 0.4 대응)."""
    scores = [macro_f1(y, probs, [g] * y.shape[1]) for g in grid]
    k = int(np.argmax(scores))
    return float(grid[k]), float(scores[k]), list(zip([float(g) for g in grid], scores))


def cv_folds(y_col, k, rng):
    """클래스별로 그 라벨에 대해 stratified k-fold 인덱스."""
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = np.where(y_col == cls)[0]
        rng.shuffle(idx)
        for j, v in enumerate(idx):
            folds[j % k].append(v)
    return [np.array(sorted(f)) for f in folds]


def main():
    args = parse_args()
    cache, out_dir = Path(args.cache_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tr_p, tr_y, _ = load_probs(cache, args.hidden, "train", args.head_file)
    dv_p, dv_y, dv_names = load_probs(cache, args.hidden, "dev", args.head_file)
    n_lab = dv_y.shape[1]
    assert n_lab == len(LABEL_COLS), f"{n_lab} != {len(LABEL_COLS)}"

    dev_macro_auroc = float(np.nanmean([auroc(dv_y[:, i], dv_p[:, i]) for i in range(n_lab)]))
    print(f"cache={cache}  head={args.head_file}")
    print(f"train n={len(tr_y)}  dev n={len(dv_y)}   dev macro AUROC={dev_macro_auroc:.4f}")
    print("  기록값 대조: original_pillar_lz4_half+probe_head=0.8693, "
          "lapa_lz4_half+finetune_head=0.8729. 어긋나면 임계값을 쓰지 말 것.\n")

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    g_thr, g_f1, g_curve = global_best_thr(dv_y, dv_p, grid)
    g_thr_tr, g_f1_tr, _ = global_best_thr(tr_y, tr_p, grid)

    per = {}
    rows = []
    for i, lab in enumerate(LABEL_COLS):
        t_tr, f1_tr = best_f1_thr(tr_y[:, i], tr_p[:, i])
        t_dv, f1_dv = best_f1_thr(dv_y[:, i], dv_p[:, i])
        y_tr_youden, _ = best_youden_thr(tr_y[:, i], tr_p[:, i])
        ppv_thr, ppv_val, ppv_n = thr_for_ppv(tr_y[:, i], tr_p[:, i], args.ppv_target)
        npv_thr, npv_val, npv_n = thr_for_npv(tr_y[:, i], tr_p[:, i], args.npv_target)

        # dev 채점
        f1_dev_at_train_thr = f1_at(dv_y[:, i], dv_p[:, i], t_tr)[0]
        f1_dev_at_global = f1_at(dv_y[:, i], dv_p[:, i], g_thr)[0]
        f1_dev_at_05 = f1_at(dv_y[:, i], dv_p[:, i], 0.5)[0]
        f1_dev_at_07 = f1_at(dv_y[:, i], dv_p[:, i], 0.7)[0]

        # dev 안에서 5-fold CV (클래스별 적합의 일반화 추정)
        folds = cv_folds(dv_y[:, i], args.folds, rng)
        tp = fp = fn = 0
        for j in range(args.folds):
            te = folds[j]
            trn = np.concatenate([folds[m] for m in range(args.folds) if m != j])
            if dv_y[trn, i].sum() == 0 or dv_y[te, i].sum() == 0:
                tp = fp = fn = -1
                break
            t, _ = best_f1_thr(dv_y[trn, i], dv_p[trn, i])
            _, a, b, c = f1_at(dv_y[te, i], dv_p[te, i], t)
            tp += a; fp += b; fn += c
        cv_f1 = float("nan") if tp < 0 else (2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)

        per[lab] = dict(
            n_pos_train=int(tr_y[:, i].sum()), n_pos_dev=int(dv_y[:, i].sum()),
            prev_dev=float(dv_y[:, i].mean()),
            dev_auroc=auroc(dv_y[:, i], dv_p[:, i]),
            thr_f1max_train=t_tr, f1_train_at_thr=f1_tr,
            thr_f1max_dev=t_dv, f1_dev_at_dev_thr=f1_dv,
            thr_youden_train=y_tr_youden,
            f1_dev_at_train_thr=f1_dev_at_train_thr,
            f1_dev_at_global_best=f1_dev_at_global,
            f1_dev_at_050=f1_dev_at_05, f1_dev_at_070=f1_dev_at_07,
            f1_dev_cv=cv_f1,
            thr_ppv_target=ppv_thr, ppv_train=ppv_val, n_flagged_train=ppv_n,
            thr_npv_target=npv_thr, npv_train=npv_val, n_neg_train=npv_n,
        )
        rows.append((lab, per[lab]))

    thrs_train_fit = [per[l]["thr_f1max_train"] for l in LABEL_COLS]
    thrs_dev_fit = [per[l]["thr_f1max_dev"] for l in LABEL_COLS]

    policies = {
        "global_0.5": macro_f1(dv_y, dv_p, [0.5] * n_lab),
        "global_0.7": macro_f1(dv_y, dv_p, [0.7] * n_lab),
        f"global_best_dev({g_thr:.2f})": g_f1,
        f"global_best_train({g_thr_tr:.2f})": macro_f1(dv_y, dv_p, [g_thr_tr] * n_lab),
        "per_class_train_fit": macro_f1(dv_y, dv_p, thrs_train_fit),
        "per_class_dev_fit(낙관 상한)": macro_f1(dv_y, dv_p, thrs_dev_fit),
        "per_class_dev_5foldCV": float(np.nanmean([per[l]["f1_dev_cv"] for l in LABEL_COLS])),
    }

    # ---- 표 --------------------------------------------------------------- #
    print(f"{'label':38s} {'n+dev':>5s} {'AUROC':>6s} | {'thr_tr':>6s} {'thr_dv':>6s} "
          f"{'thr_You':>7s} | {'F1@0.5':>6s} {'F1@0.7':>6s} {'F1@g':>6s} {'F1@tr':>6s} "
          f"{'F1@dv':>6s} {'F1cv':>6s}")
    print("-" * 128)
    for lab, d in rows:
        print(f"{lab:38s} {d['n_pos_dev']:5d} {d['dev_auroc']:6.3f} | "
              f"{d['thr_f1max_train']:6.3f} {d['thr_f1max_dev']:6.3f} {d['thr_youden_train']:7.3f} | "
              f"{d['f1_dev_at_050']:6.3f} {d['f1_dev_at_070']:6.3f} {d['f1_dev_at_global_best']:6.3f} "
              f"{d['f1_dev_at_train_thr']:6.3f} {d['f1_dev_at_dev_thr']:6.3f} {d['f1_dev_cv']:6.3f}")
    print("-" * 128)
    print("\n[dev macro-F1: 정책 비교]")
    for k, v in policies.items():
        print(f"  {k:34s} {v:.4f}")

    print(f"\n[전역 단일 임계값 sweep (dev macro-F1)]  argmax={g_thr:.2f} -> {g_f1:.4f}")
    print("  " + "  ".join(f"{t:.2f}:{s:.4f}" for t, s in g_curve if abs(round(t * 100) % 5) < 1e-9))

    # ---- paired bootstrap: per-class(train fit) vs global best(train fit) --- #
    boot = None
    if args.boot > 0:
        n = len(dv_y)
        a_thrs, b_thrs = thrs_train_fit, [g_thr_tr] * n_lab
        diffs = []
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            diffs.append(macro_f1(dv_y[idx], dv_p[idx], a_thrs) - macro_f1(dv_y[idx], dv_p[idx], b_thrs))
        diffs = np.array(diffs)
        boot = dict(n_boot=args.boot, mean=float(diffs.mean()),
                    ci_lo=float(np.percentile(diffs, 2.5)), ci_hi=float(np.percentile(diffs, 97.5)),
                    p_win=float((diffs > 0).mean()))
        print(f"\n[paired bootstrap {args.boot}회] per_class(train fit) - global_best(train fit)")
        print(f"  Δmacro-F1 = {boot['mean']:+.4f}  95% CI [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]  "
              f"P(>0)={boot['p_win']:.3f}")

    out = dict(cache_dir=str(cache), hidden=args.hidden, dev_macro_auroc=dev_macro_auroc,
               ppv_target=args.ppv_target, npv_target=args.npv_target,
               global_best_thr_dev=g_thr, global_best_thr_train=g_thr_tr,
               global_sweep_dev=g_curve, dev_macro_f1_by_policy=policies,
               bootstrap=boot,
               thresholds_f1max_train={l: per[l]["thr_f1max_train"] for l in LABEL_COLS},
               thresholds_f1max_dev={l: per[l]["thr_f1max_dev"] for l in LABEL_COLS},
               thresholds_ppv_target={l: per[l]["thr_ppv_target"] for l in LABEL_COLS},
               thresholds_npv_target={l: per[l]["thr_npv_target"] for l in LABEL_COLS},
               per_class=per)
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n-> {out_dir/'thresholds.json'}")


if __name__ == "__main__":
    main()


########## 실행 기록

# uv run --no-project --with torch --with numpy python scripts/tune_abn_thresholds.py
#
# 오픈소스 probe (dev macro AUROC 0.8693) — 전역 최적 0.64 / 클래스별 dev macro-F1 0.6046
# python3 scripts/tune_abn_thresholds.py
#
# abn 1등 lapa (dev 0.8726) — 전역 최적 0.65 / 클래스별 dev macro-F1 0.6123
# python3 scripts/tune_abn_thresholds.py \
#     --cache-dir /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-pretrain/data/abn_probe_cache/lapa_lz4_half \
#     --head-file finetune_head.pt \
#     --output-dir logs/CTRATE/CTRATE_ABN_threshold_tuning_lapa
