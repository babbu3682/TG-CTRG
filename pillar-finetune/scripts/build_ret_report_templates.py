#!/usr/bin/env python3
"""
retrieval 예측(`ret_inference.py --report-output`)을 **CTRG-3D의 `ret_template_path`가
먹는 형태**로 바꾼다. 하는 일은 하나뿐이다: 채택된 후처리(pp = display_overrides 표면형
치환 + 치환으로 생긴 중복 제거)를 `pred_key_phrases`에 적용해서 다시 쓴다.

왜 별도 산출물인가
------------------
`ret_reports_to_reportgen_csv.py`는 pp를 **채점용 CSV를 만들 때만** 적용한다. 즉
`dev_ret_reports.json` 안의 문장은 여전히 치환 전 문구이고, 그 파일을 그대로 CTRG-3D에
넘기면 **CRG 0.5322를 만든 문장이 아니라 0.5229를 만든 문장**이 프롬프트로 들어간다.
치환의 이득(Cardiomegaly F1 0.000 -> 0.587)이 LLM 입력에서 사라지는 셈이다.
→ docs/peer-transq-comparison.md 부록 A, docs/current-best-settings.md

치환 로직은 `ret_reports_to_reportgen_csv.apply_display_overrides`를 **그대로 import**해서
쓴다. 복사하면 채점 경로와 조용히 어긋날 수 있다.

Usage:
    python scripts/build_ret_report_templates.py \
        --reports-json logs/.../train_ret_reports.json \
        --display-overrides data/display_overrides.json \
        --out-json logs/.../templates_ep32_pp/train_ret_reports_pp.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ret_reports_to_reportgen_csv import apply_display_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-json", required=True,
                    help="ret_inference.py --report-output JSON (치환 전)")
    ap.add_argument("--display-overrides", required=True,
                    help="JSON {문장: 대체 문구}. pp 없는 템플릿이 필요하면 원본 파일을 "
                         "그대로 쓰면 되므로 이 인자는 필수로 둔다.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--split-json", default=None,
                    help="선택. 주면 accession 집합이 이 split과 정확히 일치하는지 검사한다.")
    args = ap.parse_args()

    with open(args.display_overrides, encoding="utf-8") as f:
        overrides = json.load(f)
    if not overrides:
        raise SystemExit(f"display_overrides가 비었다: {args.display_overrides}")

    records = json.load(open(args.reports_json, encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{args.reports_json}는 비어 있지 않은 list여야 한다.")

    out, n_hit, n_before, n_after, n_empty = [], 0, 0, 0, []
    seen = set()
    for rec in records:
        acc = rec.get("accession")
        if acc is None:
            raise SystemExit(f"accession이 없는 레코드가 있다: {list(rec.keys())}")
        if acc in seen:
            raise SystemExit(f"accession 중복: {acc}")
        seen.add(acc)

        raw = rec.get("pred_key_phrases")
        if raw is None:
            raise SystemExit(f"{acc}에 pred_key_phrases가 없다 (keys={list(rec.keys())})")

        hit = any((p or "").strip().rstrip(".").strip() in overrides for p in raw)
        n_hit += int(hit)
        phrases = apply_display_overrides(raw, overrides)
        n_before += len(raw)
        n_after += len(phrases)
        if not phrases:
            n_empty.append(acc)

        new = {"accession": acc, "pred_key_phrases": phrases}
        for k in ("pred_threshold", "pred_num_fired"):
            if k in rec:
                new[k] = rec[k]
        new["pred_num_phrases_before_pp"] = len(raw)
        out.append(new)

    if args.split_json:
        want = set()
        with open(args.split_json, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    want.add(json.loads(line)["sample_name"])
        if want != seen:
            raise SystemExit(
                f"accession 집합 불일치: 예측에만 {len(seen - want)}개, split에만 "
                f"{len(want - seen)}개 (예: {sorted(want - seen)[:5]})")
        print(f"  split 정합: {len(seen)}/{len(want)} 완전 일치")

    dst = Path(args.out_json)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    n = len(out)
    print(f"wrote {n} records -> {dst}")
    print(f"  overrides            : {len(overrides)}개")
    print(f"  치환이 걸린 exam     : {n_hit}/{n}")
    print(f"  문장/exam            : {n_before / n:.2f} -> {n_after / n:.2f} (치환 후 중복 제거)")
    if n_empty:
        print(f"  [warn] 문장 0개인 exam {len(n_empty)}개: {n_empty[:5]}")


if __name__ == "__main__":
    main()
