#!/usr/bin/env python3
"""
Turn scripts/ret_inference.py's retrieved-phrase JSON into the result CSV that
PETRG-3D/src/evaluate_reportgen.py (the VLM3D challenge scorer wrapper) eats.

The retrieval head emits a *set of sentences* per exam, not a paragraph, so the
report has to be assembled here. We join the retrieved phrases with ". " and
close with a final period -- i.e. exactly "sentence1. sentence2." -- because
that is the form the challenge's two consumers both expect:
  * nlg_metrics.py tokenizes plain text, and its own normalization collapses
    " . " / " ." into "." -- so sentence-terminating periods must be present or
    BLEU sees one giant run-on sentence.
  * the RadBERT labeler (infer.py) was trained on real report prose.

Deliberately NOT prefixed with "Findings: ", even though the reference reports
(and the CTRG-3D LLM's generations, which learned the format) all start with it.
The retrieval output is a bag of findings sentences and the requested format was
the bare concatenation; adding the prefix would put two extra tokens per exam
into BLEU that the model never predicted. Pass --prefix "Findings: " to match
the reference format instead.

GT_report comes from the split json's `report_metadata`, verified byte-identical
(1552/1552) to the ground_truth.json used to score the CTRG-3D checkpoints, so
BLEU here is directly comparable to those numbers.

Usage:
    python scripts/ret_reports_to_reportgen_csv.py \
        --reports-json  logs/.../dev_ret_reports.json \
        --split-json    /data/.../merged_valid_filtered_prob_half.json \
        --out-csv       logs/.../dev_reportgen_input.csv
"""

import argparse
import csv
import json
from pathlib import Path


def join_phrases(phrases, prefix="", overrides=None):
    """['a', 'b.'] -> 'a. b.' (with prefix prepended if given).

    overrides: {문장: 대체 문구}. emit 직전에 표면형만 치환한다 — **검색 결과 자체는
        건드리지 않는다**. gallery 임베딩도, 어떤 문장이 뽑혔는지도 그대로고 바뀌는 것은
        출력 텍스트뿐이다.

        존재 이유는 채점기 쪽에 있다: 문장 타깃을 만든 LLM 정규화가 일부 소견을 맨명사로
        뭉갰는데("Heart size increased" -> "Cardiomegaly"), 정상 소견 문장 여러 개 뒤에
        붙은 맨명사를 RadBERT labeler가 거의 못 잡는다. ep32 채점에서 **Cardiomegaly만
        P/R/F1이 전부 0.000**이었는데 dev 1,552례 중 295례가 그 문구를 실제로 뽑고 있었고,
        치환 후 **F1 0.000 -> 0.587 / macro-F1 +0.0319 / CRG +0.0093**이 됐다
        (job 28030540, docs/peer-transq-comparison.md 부록 A).

        gallery 파일을 직접 고치지 않는 이유: recall@k가 gallery 문자열로 GT를 조회하므로
        원본을 바꾸면 검색 지표가 깨진다. 치환 후 이미 뽑힌 문구와 충돌할 수 있어 다시
        중복 제거한다.

    (같은 job에서 "학습 리포트에서의 평균 위치로 문장을 재정렬"하는 변형도 같이 쟀지만
    CRG −0.0056 / macro-F1 −0.0039로 **기각**했다. ROUGE_L만 +0.0038 올랐는데 헤드라인
    지표가 아니다. 그 코드 경로는 제거했다 — 되살릴 일이 생기면 문장별 평균 위치
    {문장: i/(n-1)}를 train split의 `sentences`에서 집계해 정렬 키로 쓰면 된다.)
    """
    cleaned = apply_display_overrides(phrases, overrides)
    if not cleaned:
        return prefix.strip()
    return (prefix + ". ".join(cleaned) + ".").strip()


def apply_display_overrides(phrases, overrides=None):
    """['a.', 'b'] -> ['a', 'b'], 그 다음 overrides가 있으면 표면형 치환 + 중복 제거.

    join_phrases()가 문장을 이어 붙이기 직전에 하는 일과 **정확히 같은 변환**이고, 리스트
    형태로 돌려준다. CTRG-3D에 넘길 ret 템플릿(`pred_key_phrases`)을 채점 경로와 한 글자도
    다르지 않게 만들기 위해 분리했다 — 두 곳에 따로 구현하면 pp를 적용한 템플릿과 pp로
    채점한 지표가 조용히 어긋난다.

    중복 제거를 overrides가 있을 때만 하는 것은 의도된 동작이다: 치환 전에는 gallery에서
    서로 다른 문장이 뽑혔다는 사실 자체가 정보이고(같은 문장이 두 번 뽑히지도 않는다),
    치환 후에는 여러 표면형이 한 문구로 뭉쳐져 실제로 중복이 생긴다.
    """
    cleaned = []
    for p in phrases:
        p = (p or "").strip().rstrip(".").strip()
        if p:
            cleaned.append(p)
    if overrides:
        cleaned = [overrides.get(s, s) for s in cleaned]
        cleaned = list(dict.fromkeys(cleaned))
    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-json", required=True,
                    help="ret_inference.py --report-output JSON")
    ap.add_argument("--split-json", required=True,
                    help="merged_*.json used as the split (for report_metadata / GT text)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--prefix", default="",
                    help='Optional report prefix, e.g. "Findings: " (default: none)')
    ap.add_argument("--display-overrides", default=None,
                    help="JSON {문장: 대체 문구}. 출력 텍스트만 바꾼다 (join_phrases 참고). "
                         "기본값은 적용 안 함 — 후처리 없는 값과 나란히 비교하기 위해서다.")
    args = ap.parse_args()

    overrides = None
    if args.display_overrides:
        with open(args.display_overrides, encoding="utf-8") as f:
            overrides = json.load(f)

    records = json.load(open(args.reports_json, encoding="utf-8"))
    gt = {}
    with open(args.split_json, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gt[r["sample_name"]] = r["report_metadata"]

    rows, missing, empty = [], [], []
    n_phrases = 0
    for rec in records:
        name = rec["accession"]
        if name not in gt:
            missing.append(name)
            continue
        phrases = rec.get("pred_key_phrases") or []
        if not phrases:
            empty.append(name)
        n_phrases += len(phrases)
        rows.append({
            "SampleName": name,
            "Question": "Describe the findings of this chest CT.",
            "GT_report": gt[name],
            "Pred_report": join_phrases(phrases, args.prefix, overrides=overrides),
        })

    if missing:
        raise SystemExit(
            f"{len(missing)} accession(s) have no report_metadata in {args.split_json}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}. Wrong split json?")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["SampleName", "Question", "GT_report", "Pred_report"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out}")
    print(f"  retrieved sentences per exam: mean {n_phrases / max(1, len(rows)):.2f}")
    if empty:
        print(f"  [warn] {len(empty)} exam(s) retrieved zero sentences: {empty[:5]}")
    print(f"  prefix: {args.prefix!r}")
    print(f"  display_overrides: {len(overrides) if overrides else 0}개")
    if overrides:
        n_hit = sum(1 for rec in records
                    if any((p or '').strip() in overrides for p in (rec.get('pred_key_phrases') or [])))
        print(f"  치환이 실제로 걸린 exam: {n_hit}/{len(rows)}")
    print(f"  sample Pred_report: {rows[0]['Pred_report'][:200]}")


if __name__ == "__main__":
    main()
