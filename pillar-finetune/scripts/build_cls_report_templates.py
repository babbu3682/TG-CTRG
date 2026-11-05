"""abn 승자 probe 모델로 train/valid split의 cls_report 템플릿 json을 생성한다.

승자 세팅(CLAUDE.md "현재 최고 결과" 참고): Pillar0-ChestCT 오픈소스 인코더(frozen) ->
pooled(3-scale maxpool+concat, 1152차원) -> MLP[1024] probe head, dev macro AUROC 0.8692.

인코더를 다시 돌리지 않는다: scripts/extract_abn_features.py가 이미 pooled feature를
abn_probe_cache/<tag>/{train,dev}.pt로 캐싱해뒀으므로, 여기서는 그 feature에 probe head만
태워 logit -> sigmoid를 계산한다 (23,869 + 1,552 샘플 전체가 CPU 몇 초). GPU 불필요.

임계값은 전역 고정(pos >= 0.7, neg <= 0.3). 그 사이 구간은 "uncertain"으로 두고 문장을
아예 생략한다 -- 확실한 문장만 LLM에 넘기기 위함. 문장 자체는 config의 report_templates에서
읽으므로 scripts/abn_inference.py가 만드는 리포트와 문구가 동일하다.

출력 json 스키마 (샘플당 1개 객체):
    {"accession": ..., "report": [문장, ...],
     "status_<label>": "positive"|"negative"|"uncertain", ...,
     "prob_<label>": float}
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# pillar 패키지 전체를 import하면 kornia -> flash_attn 경로가 컨테이너 밖에서 GLIBC 에러를
# 내므로, 이 스크립트가 필요로 하는 라벨 순서만 pillar/metrics/abn_metric.py 정의 그대로 둔다.
# (pred 차원 순서 = 이 순서. 바뀌면 abn_metric.py와 함께 고칠 것.)
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
DEFAULT_CONFIG = "configs/ctrate_abnclass_finetune_lz4_pillar_opensource.yaml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE,
                   help="extract_abn_features.py가 만든 {train,dev}.pt + probe_head.pt가 있는 디렉토리")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="report_templates 문장을 읽어올 config")
    p.add_argument("--head-file", default="probe_head.pt",
                   help="cache-dir 안의 head state_dict 파일 이름. probe 계열은 "
                        "probe_head.pt(Sequential 인덱스 0/3), full-finetune 체크포인트에서 "
                        "뽑은 head는 finetune_head.pt처럼 별도 이름을 쓴다.")
    p.add_argument("--hidden", type=int, default=1024, help="probe head의 hidden 차원 (승자는 1024)")
    p.add_argument("--pos-threshold", type=float, default=0.7,
                   help="전역 positive 임계값. --thresholds-json을 주면 무시됨")
    p.add_argument("--neg-threshold", type=float, default=0.3,
                   help="전역 negative 임계값. --thresholds-json을 주면 무시됨")
    p.add_argument("--thresholds-json", default=None,
                   help='클래스별 임계값 json. {"positive": {label: thr, ...}, '
                        '"negative": {label: thr, ...}} 형태. "negative"를 생략하면 '
                        'negative = positive로 두어 uncertain 구간 없이 18칸을 모두 채운다. '
                        'scripts/tune_abn_thresholds.py가 만드는 thresholds.json의 '
                        'thresholds_f1max_train 블록을 그대로 쓸 수 있다.')
    p.add_argument("--splits", nargs="+", default=["train", "dev"],
                   help="캐시 파일 이름 기준. dev.pt가 valid(half) split이다.")
    p.add_argument("--output-dir", default="logs/CTRATE/CTRATE_ABN_cls_report_templates")
    return p.parse_args()


def load_sentences(config_path):
    """config의 report_templates에서 라벨별 positive/negative 문장을 LABEL_COLS 순서로 뽑는다."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    rt = cfg.get("report_templates")
    if rt is None:
        raise ValueError(f"{config_path}에 report_templates 섹션이 없습니다.")
    by_name = {e["name"]: e for e in rt["labels"]}
    missing = [c for c in LABEL_COLS if c not in by_name]
    if missing:
        raise ValueError(f"report_templates에 없는 라벨: {missing}")
    return [(by_name[c]["positive"], by_name[c]["negative"]) for c in LABEL_COLS]


def load_thresholds(path):
    """클래스별 임계값 json -> (pos[18], neg[18]) numpy 배열.

    18개 라벨이 하나라도 빠지면 즉시 raise한다 -- 빠진 라벨을 전역값으로 조용히 메우면
    "이 클래스는 성능이 나쁘다"는 잘못된 결론으로 직행하기 때문.
    """
    with open(path) as f:
        blob = json.load(f)
    if "positive" not in blob:
        raise ValueError(f'{path}에 "positive" 키가 없습니다. 키: {sorted(blob)}')
    pos_map = blob["positive"]
    neg_map = blob.get("negative", pos_map)   # 생략 시 uncertain 구간 없음
    for name, m in (("positive", pos_map), ("negative", neg_map)):
        missing = [c for c in LABEL_COLS if c not in m]
        if missing:
            raise ValueError(f'{path}의 "{name}"에 없는 라벨: {missing}')
        bad = [c for c in LABEL_COLS if m[c] is None]
        if bad:
            raise ValueError(f'{path}의 "{name}"에서 임계값이 null인 라벨: {bad} '
                             "(목표 PPV/NPV를 만족하는 임계값이 없었던 클래스)")
    pos = np.array([float(pos_map[c]) for c in LABEL_COLS])
    neg = np.array([float(neg_map[c]) for c in LABEL_COLS])
    bad = [LABEL_COLS[i] for i in range(len(LABEL_COLS)) if neg[i] > pos[i]]
    if bad:
        raise ValueError(f"negative 임계값이 positive보다 큰 라벨: {bad}")
    return pos, neg


def build_head(state, hidden, in_dim, n_labels):
    """train_abn_probe.py와 동일 구조. eval 모드이므로 Dropout은 항등."""
    head = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.0), nn.Linear(hidden, n_labels)
    )
    head.load_state_dict(state)
    head.eval()
    return head


def auroc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 동점은 평균 rank로
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    args = parse_args()
    cache = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.thresholds_json:
        pos_thr, neg_thr = load_thresholds(args.thresholds_json)
    else:
        if args.neg_threshold >= args.pos_threshold:
            raise ValueError(f"neg_threshold({args.neg_threshold})는 pos_threshold({args.pos_threshold})보다 "
                             "작아야 그 사이에 uncertain 구간이 생깁니다.")
        pos_thr = np.full(len(LABEL_COLS), args.pos_threshold)
        neg_thr = np.full(len(LABEL_COLS), args.neg_threshold)

    sentences = load_sentences(args.config)
    head_state = torch.load(cache / args.head_file, map_location="cpu", weights_only=False)
    print(f"head      : {cache/args.head_file}  hidden={args.hidden}")
    print(f"문장 출처 : {args.config}")
    if args.thresholds_json:
        print(f"임계값    : 클래스별 ({args.thresholds_json})")
        for i, label in enumerate(LABEL_COLS):
            band = "" if pos_thr[i] == neg_thr[i] else "  (그 사이는 문장 생략)"
            print(f"            {label:38s} pos>={pos_thr[i]:.3f}  neg<={neg_thr[i]:.3f}{band}")
        print()
    else:
        print(f"임계값    : positive >= {args.pos_threshold}, negative <= {args.neg_threshold} "
              f"(그 사이는 문장 생략)\n")

    summary = {}
    for split in args.splits:
        blob = torch.load(cache / f"{split}.pt", map_location="cpu", weights_only=False)
        X, y, names = blob["features"], blob["labels"], blob["sample_names"]
        head = build_head(head_state, args.hidden, X.shape[1], y.shape[1])
        with torch.no_grad():
            probs = torch.sigmoid(head(X)).numpy()
        y = y.numpy()

        macro = float(np.nanmean([auroc(y[:, i], probs[:, i]) for i in range(y.shape[1])]))

        # pos_thr/neg_thr는 라벨축(18) 벡터 -- (n,18) 확률에 열별로 브로드캐스팅된다.
        pos_mask, neg_mask = probs >= pos_thr[None, :], probs <= neg_thr[None, :]
        ppv = float(y[pos_mask].mean()) if pos_mask.any() else float("nan")
        npv = float((1 - y[neg_mask]).mean()) if neg_mask.any() else float("nan")
        cov = float((pos_mask | neg_mask).mean())

        per_class = {}
        for i, label in enumerate(LABEL_COLS):
            pm, nm = pos_mask[:, i], neg_mask[:, i]
            tp = int((pm & (y[:, i] == 1)).sum()); fp = int((pm & (y[:, i] == 0)).sum())
            fn = int((~pm & (y[:, i] == 1)).sum())
            per_class[label] = dict(
                pos_threshold=float(pos_thr[i]), neg_threshold=float(neg_thr[i]),
                n_positive_cells=int(pm.sum()), n_negative_cells=int(nm.sum()),
                ppv=float(y[pm, i].mean()) if pm.any() else float("nan"),
                npv=float((1 - y[nm, i]).mean()) if nm.any() else float("nan"),
                f1=(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
            )

        rows = []
        for r in range(len(names)):
            row = OrderedDict(accession=str(names[r]), report=[])
            for i, label in enumerate(LABEL_COLS):
                if pos_mask[r, i]:
                    row["report"].append(sentences[i][0])
                    row[f"status_{label}"] = "positive"
                elif neg_mask[r, i]:
                    row["report"].append(sentences[i][1])
                    row[f"status_{label}"] = "negative"
                else:
                    row[f"status_{label}"] = "uncertain"
            for i, label in enumerate(LABEL_COLS):
                row[f"prob_{label}"] = round(float(probs[r, i]), 6)
            rows.append(row)

        out_path = out_dir / f"cls_report_{split}.json"
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        n_sent = float(np.mean([len(r["report"]) for r in rows]))
        macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
        summary[split] = dict(n_samples=len(rows), macro_auroc=macro, macro_f1=macro_f1,
                              ppv=ppv, npv=npv, cell_coverage=cov, mean_sentences=n_sent,
                              per_class=per_class, output=str(out_path))
        print(f"[{split}] n={len(rows)}  macro_auroc={macro:.4f}  macro_f1={macro_f1:.4f}")
        print(f"        positive 문장 PPV={ppv:.4f} ({int(pos_mask.sum())} cells)   "
              f"negative 문장 NPV={npv:.4f} ({int(neg_mask.sum())} cells)")
        print(f"        셀 커버리지={cov*100:.1f}%   샘플당 평균 문장 {n_sent:.1f}/18")
        print(f"        -> {out_path}\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(dict(config=args.config, cache_dir=str(cache), hidden=args.hidden,
                       thresholds_json=args.thresholds_json,
                       pos_threshold=None if args.thresholds_json else args.pos_threshold,
                       neg_threshold=None if args.thresholds_json else args.neg_threshold,
                       per_class_thresholds={c: dict(positive=float(pos_thr[i]),
                                                     negative=float(neg_thr[i]))
                                             for i, c in enumerate(LABEL_COLS)},
                       splits=summary), f, indent=2)
    print(f"요약 -> {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()


########## 실행 기록

# 승자 probe(Pillar0-ChestCT, pooled+MLP[1024], dev macro AUROC 0.8692)로 train/valid 템플릿 생성.
# 캐시된 feature만 쓰므로 GPU 불필요, 전체 몇 초.
# uv run --no-project --with torch --with numpy --with pyyaml python scripts/build_cls_report_templates.py
#
# 임계값을 바꿔보고 싶을 때:
# uv run ... python scripts/build_cls_report_templates.py --pos-threshold 0.8 --neg-threshold 0.2
#
# 클래스별 임계값 (scripts/tune_abn_thresholds.py 산출):
#   F1 최대(uncertain 구간 없음)
# uv run ... python scripts/build_cls_report_templates.py \
#     --thresholds-json data/abn_thresholds_f1max_train.json
#   목표 PPV 0.90 / NPV 0.98 (고신뢰 문장만, 희소 클래스 positive는 거의 안 나옴)
# uv run ... python scripts/build_cls_report_templates.py \
#     --thresholds-json data/abn_thresholds_ppv90_npv98.json
#
# 최종 산출물 (2026-08-21): abn 1등 lapa + 클래스별 F1 최대 임계값.
# dev macro-F1 0.6123 / train 0.6275, uncertain 0셀(18칸 전부 채움).
# python3 scripts/build_cls_report_templates.py \
#     --cache-dir /data/drdcad/sunggu/0.Challenge/VLM3D_MICCAI/pillar-pretrain/data/abn_probe_cache/lapa_lz4_half \
#     --head-file finetune_head.pt \
#     --config configs/ctrate_abnclass_finetune_lz4_pillar_self_lapa_maxpool.yaml \
#     --thresholds-json data/abn_thresholds_f1max_train_lapa.json \
#     --splits train dev \
#     --output-dir logs/CTRATE/CTRATE_ABN_cls_report_templates_lapa_f1max
