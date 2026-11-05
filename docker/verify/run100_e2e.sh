#!/usr/bin/env bash
# ==============================================================================
# End-to-end reproduction on 100 valid exams, through the container's OWN
# predict.py — the same entry point the platform runs, only with /input,
# /output, /checkpoint and /weights pointed at local directories.
#
# This is the decisive pre-submission check. It exercises every stage
# (A, A', B x 10 draws, C RadBERT, D merge, E verify loop) and then scores the
# result with the official scorer, paired against `bb1000F5I0_v` restricted to
# the same 100 exams.
#
#   sbatch verify/run100_e2e.sbatch
#
# Expected: CRG in the 0.53-0.57 band. The full-valid reference is 0.5749; a
# 100-exam subset moves it around, which is why the comparison below is PAIRED
# (same exams, same scorer, same GT) rather than against the headline number.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
DOCKER=$PWD
PT=/vf/users/drdcad/sunggu/0.Challenge/PETRG-3D
PY=$PT/.venv/bin/python
MAPPING=/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/ctrate/ctrate_cache_1.25mm_lz4/valid/mapping_path.csv
SPLIT_JSON=/data/drdcad/sunggu/0.Challenge/pillar-pretrain/data/merged_valid_filtered_prob_half.json
RADBERT_CKPT=$PT/data/radbert/RadBertClassifier.pth
RADBERT_DIR=/vf/users/drdcad/sunggu/0.Challenge/VLM3D-Dockers/ct_challenges/reportgen_evaluation/assets/radbert_local
REF_CSV=$PT/results/grpo_newencB/bb1000F5I0_v.csv

N=${N:-100}
RUN=${RUN:-$DOCKER/verify/run100}
rm -rf "$RUN/input" "$RUN/output"
mkdir -p "$RUN/input" "$RUN/output" "$RUN/checkpoint"

echo "=== staging $N exams into $RUN/input ==="
"$PY" - "$MAPPING" "$REF_CSV" "$RUN/input" "$N" <<'EOF'
import csv, os, sys
mapping, ref_csv, dst, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
# Take the first N exams the reference CSV covers, in its own row order, so the
# paired comparison uses exactly the same set.
import pandas as pd
names = pd.read_csv(ref_csv)["SampleName"].tolist()[:n]
src = {r["sample_name"]: r["source_path"] for r in csv.DictReader(open(mapping))}
missing = [x for x in names if x not in src]
assert not missing, f"no source nii for {missing[:3]}"
for x in names:
    os.symlink(src[x], os.path.join(dst, f"{x}.nii.gz"))
print(f"  linked {len(names)} volumes; first={names[0]} last={names[-1]}")
EOF

echo "=== running predict.py (container entry point) ==="
cd "$DOCKER"
PYTHONPATH="$DOCKER/src:$DOCKER/src/vendor:$DOCKER/src/vendor/petrg:$DOCKER/src/vendor/rad-vision-engine" \
INPUT_DIR="$RUN/input" OUTPUT_DIR="$RUN/output" CHECKPOINT_DIR="$RUN/checkpoint" \
WEIGHTS_DIR="$DOCKER/weights" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" predict.py

echo "=== predictions.json -> scorer CSV ==="
"$PY" - "$RUN/output/predictions.json" "$REF_CSV" "$RUN/pred.csv" <<'EOF'
import json, sys
import pandas as pd
pred_json, ref_csv, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
blob = json.load(open(pred_json))
items = blob[0]["outputs"][0]["value"]["generated_reports"]
got = {it["input_image_name"]: it["report"] for it in items}
ref = pd.read_csv(ref_csv).set_index("SampleName")
rows = [{"SampleName": n, "Question": ref.loc[n, "Question"],
         "GT_report": ref.loc[n, "GT_report"], "Pred_report": got[n]}
        for n in got]
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"  {len(rows)} rows -> {out_csv}")
EOF

echo "=== scoring: container run ==="
cd "$PT"
"$PY" src/evaluate_reportgen.py --result_csv "$RUN/pred.csv" \
    --split_json_path "$SPLIT_JSON" --out_dir "$RUN/eval_container" \
    --radbert_model_path "$RADBERT_CKPT" --radbert_dir "$RADBERT_DIR"

echo "=== scoring: reference bb1000F5I0_v on the SAME exams ==="
"$PY" - "$REF_CSV" "$RUN/pred.csv" "$RUN/ref_subset.csv" <<'EOF'
import sys
import pandas as pd
ref, mine, out = sys.argv[1], sys.argv[2], sys.argv[3]
names = pd.read_csv(mine)["SampleName"].tolist()
pd.read_csv(ref).set_index("SampleName").loc[names].reset_index().to_csv(out, index=False)
print(f"  {len(names)} rows -> {out}")
EOF
"$PY" src/evaluate_reportgen.py --result_csv "$RUN/ref_subset.csv" \
    --split_json_path "$SPLIT_JSON" --out_dir "$RUN/eval_reference" \
    --radbert_model_path "$RADBERT_CKPT" --radbert_dir "$RADBERT_DIR"

echo "=== paired comparison ==="
"$PY" - "$RUN/eval_container/metrics.json" "$RUN/eval_reference/metrics.json" <<'EOF'
import json, sys
def get(p):
    m = json.load(open(p))
    return (m["crg"]["CRG"], m["classification"]["macro"]["f1"],
            m["generation"]["BLEU_mean"])
c = get(sys.argv[1]); r = get(sys.argv[2])
print(f"{'':14s} {'CRG':>8s} {'macroF1':>8s} {'BLEU':>8s} {'sum':>8s}")
for tag, v in (("container", c), ("reference", r)):
    print(f"{tag:14s} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.5f} {sum(v):8.4f}")
d = [a - b for a, b in zip(c, r)]
print(f"{'delta':14s} {d[0]:+8.4f} {d[1]:+8.4f} {d[2]:+8.5f} {sum(d):+8.4f}")
print("\nsignificance floor (same policy, scored twice): "
      "|dCRG|<0.008  |dF1|<0.013  |dBLEU|<0.007")
EOF
