# CTRG-3D with GRPO post-training and label-vote ensemble

2nd place, VLM3D Challenge 2026 — CT Report Generation (MICCAI 2026).

Chest CT volume → radiology report (Findings / Impression), trained on CT-RATE.
Hidden test set: CRG 0.5562, macro-F1 0.6011, BLEU 0.2646.

## Overview

<p align="center"><img src="assets/pipeline.png" width="820"></p>

| | component | code |
|---|---|---|
| 1 | Preprocessing: 1.25 mm resampling, 256³, lossless LZ4 cache; sentence-level labels; Qwen3-Embedding-8B sentence embeddings | `rad-vision-engine/`, `pillar-pretrain/scripts/` |
| 2 | 3D encoder: Pillar-0 continued-pretrained with a phrase-level vision–language objective (VL-CABS) | `pillar-pretrain/` |
| 3 | CLS: 18-finding probe on the frozen encoder → 18 template sentences per study | `pillar-finetune/` |
| 4 | RET: DETR-style retrieval head → sentences retrieved from a 79k-sentence gallery | `pillar-finetune/` |
| 5 | MLLM: frozen encoder → Perceiver (128 tokens) → Qwen3-8B + LoRA, prompted with the CLS and RET templates; SFT with finding-weighted tokens, then GRPO post-training; 10-draw label-vote ensemble | `CTRG-3D/` |
| 6 | Inference container (runs 1–5 end to end per volume) | `docker/` |

Validation split (1,552 studies, official scorer):

| | CRG | macro-F1 | BLEU |
|---|---|---|---|
| SFT, public Pillar0-ChestCT encoder | 0.446 | 0.430 | 0.321 |
| + continued-pretrained encoder | 0.466 | 0.471 | 0.318 |
| + finding-weighted SFT loss | 0.505 | 0.531 | 0.326 |
| + GRPO (single policy) | 0.547 | 0.597 | 0.380 |
| + 10-draw ensemble (submitted) | 0.580 | 0.619 | 0.373 |

## Pretrained weights

All checkpoints are hosted on Hugging Face: **`<HF_REPO>`** (≈21 GB; Qwen3-8B is fetched separately).

```bash
pip install -U "huggingface_hub[cli]"
bash scripts/hf_download_weights.sh <HF_REPO> docker/weights
```

| file | component |
|---|---|
| `lapa_epoch12.pt` | encoder (stage 2) |
| `cls_finetune_head.pt`, `abn_thresholds_f1max_train_lapa.json` | CLS probe (stage 3) |
| `ret_head.pt`, `gallery_embeddings_lasthidden.pt`, `gallery_sentences.pt` | RET head and gallery (stage 4) |
| `ctrg3d/model.safetensors` | MLLM after SFT (stage 5) |
| `lora/*.safetensors` | GRPO LoRA adapters (9 shipped; the ensemble uses 8) (stage 5) |
| `radbert/` | RadBERT labeler from the challenge evaluation kit |

Qwen3-8B is downloaded separately from its own repository.

## Installation

Each subdirectory keeps its own environment (`pyproject.toml` / `requirements.txt`).
Training was done with Python 3.10, PyTorch 2.13 (CUDA 13), transformers 4.57, peft 0.17,
trl 1.10 and DeepSpeed 0.19 on A100-80GB GPUs. The inference container uses PyTorch 2.8 (CUDA 12.6).

Paths to data, checkpoints and base models are set at the top of each launcher and config;
edit them for your environment.

## Pipeline

### 1. Preprocessing

```bash
cd rad-vision-engine && pip install -e .
vision-engine process --config configs/ct_chest.yaml \
    --input-series-csv <csv of CT-RATE *_1 series> --output <cache>/train --workers 32
```
Builds the LZ4 volume cache (1.25 mm isotropic, central 256 slices, 256×256). Repeat for `valid`.
The split files `merged_{train,valid}_filtered_prob_half.json` hold, per study, the report,
its sentences, per-sentence 18-finding labels and the official study-level labels.

```bash
cd pillar-pretrain
python scripts/cache_ctrate_embeddings_phrase_align.py --model-name <Qwen3-Embedding-8B> \
    --json-file data/merged_train_filtered_prob_half.json \
    --cache-dir data/text_cache_qwen3_embedding_8b_phrase/train
```

### 2. Encoder pretraining

```bash
cd pillar-pretrain
bash train_unimodal_pillar0_ctrate_chest_ct_1node_lapa.sh   # 2×A100, ~2 days
```
Starts from the public `Pillar0-ChestCT` weights. Objective: each report sentence attends over the
volume's 1,024 spatial tokens and is contrasted against its source volume, with finding labels
removing same-finding volumes from the negatives; no global image–report term. The checkpoint
with the lowest validation loss (epoch 12) is used frozen everywhere below.

### 3. CLS probe and template

```bash
cd pillar-finetune
bash scripts/run_ctrate_abnclass.sh configs/ctrate_abnclass_finetune_lz4_pillar_self_lapa_maxpool.yaml \
     CTRATE_FT_Atlas_abn_bce_self_lapa_maxpool
python scripts/extract_abn_features.py configs/ctrate_abnclass_finetune_lz4_pillar_self_lapa_maxpool.yaml \
     --tag lapa_lz4_half --splits dev,train
python scripts/tune_abn_thresholds.py --cache-dir <cache>/lapa_lz4_half --head-file finetune_head.pt \
     --output-dir logs/CTRATE/CTRATE_ABN_threshold_tuning_lapa
python scripts/build_cls_report_templates.py --cache-dir <cache>/lapa_lz4_half --head-file finetune_head.pt \
     --config configs/ctrate_abnclass_finetune_lz4_pillar_self_lapa_maxpool.yaml \
     --thresholds-json data/abn_thresholds_f1max_train_lapa.json --splits train dev \
     --output-dir logs/CTRATE/CTRATE_ABN_cls_report_templates_lapa_f1max
```
An MLP probe on max-pooled encoder features predicts the 18 CT-RATE findings; per-class
thresholds (F1-max on train) turn the probabilities into one positive or negative sentence per
finding (`cls_report_{train,dev}.json`).

### 4. RET head and template

```bash
cd pillar-finetune
bash scripts/run_ctrate_retrieval.sh configs/ctrate_retrieval_finetune_lz4_pillar_self_lapa.yaml \
     CTRATE_FT_Atlas_retrieval_self_lapa
python scripts/build_gallery_lasttoken.py                       # 79,166 training sentences -> embeddings
python scripts/ret_inference.py configs/ctrate_retrieval_finetune_lz4_pillar_self_lapa.yaml \
     --ckpt logs/CTRATE/CTRATE_FT_Atlas_retrieval_self_lapa/checkpoints/best.ckpt \
     --split train --threshold 0.4 --report-output <out>/train_ret_reports.json      # and --split dev
python scripts/build_ret_report_templates.py --reports-json <out>/train_ret_reports.json \
     --display-overrides data/display_overrides.json \
     --out-json logs/CTRATE/CTRATE_FT_Atlas_retrieval_self_lapa/templates_ep32_pp/train_ret_reports_pp.json \
     --split-json <pillar-pretrain>/data/merged_train_filtered_prob_half.json
```
A transformer decoder with 64 queries maps the frozen features into the sentence-embedding space;
sentences whose cosine similarity exceeds 0.4 are retrieved from the gallery and lightly
normalised (`*_ret_reports_pp.json`).

### 5. MLLM: SFT → GRPO → ensemble

```bash
cd CTRG-3D
# finding-level token weights from sentence spans
python scripts/build_gt_sentence_weights.py --split_json <pillar-pretrain>/data/merged_train_filtered_prob_half.json \
     --gt_spans GT_report_sentence_label/train_sentence_token_labels.json \
     --gt_sentences GT_report_sentence_label/train_sentences_18labels.json \
     --out data/sentence_labels/train_gt_sentence_labels.json \
     --filtered_split_out data/sentence_labels/merged_train_gtcov.json        # and valid
# SFT (2×A100, LoRA r=8, 128 visual tokens, label-token weight 5, 15 epochs, lr 5e-5)
cd scripts && LANG_ENCODER_PATH=<Qwen3-8B> LEARNING_RATE=5e-5 NUM_TRAIN_EPOCHS=15 \
     bash train.sh qwen3_joint_template_new_encoder_template_gtweight
# cache the frozen visual features, build the GRPO reference (BLEU floors, F1 corpus, reward scales)
bash ../launchers/run_sbatch_cache_newencB
bash ../launchers/run_sbatch_grpo_ref_newencB
# GRPO (reward = alpha*CRG_exam + gamma*dF1 - lambda*BLEU hinge); the ensemble uses four runs
python grpo_newencB.py --ckpt <sft> --cache_dir <cache> --output_dir ../outputs/grpo_newencB      --alpha 1 --gamma 1 --lambda_bleu 1
python grpo_newencB.py --ckpt <sft> --cache_dir <cache> --output_dir ../outputs/grpo_B_a2from700  --alpha 2 --gamma 1 --lambda_bleu 1 --resume_from_checkpoint ../outputs/grpo_newencB/checkpoint-700
python grpo_newencB.py --ckpt <sft> --cache_dir <cache> --output_dir ../outputs/grpo_B_g3         --alpha 1 --gamma 3 --lambda_bleu 0
python grpo_newencB.py --ckpt <sft> --cache_dir <cache> --output_dir ../outputs/grpo_B_nobleu     --alpha 1 --gamma 1 --lambda_bleu 0
# draws (one CSV per adapter/sampler) and their RadBERT labels
python generate_batched.py --ckpt <sft> --cache_dir <cache> --split valid_newenc --adapter ../outputs/<run>/checkpoint-<n> --out ../results/<run>/<tag>.csv
python ../src/evaluate_reportgen.py --result_csv ../results/<run>/<tag>.csv --out_dir ../results/<run>/<tag>/vlm3d_eval ...
# label-vote ensemble (order of draws matters; draw 6 is the skeleton)
D="grpo_newencB/700 grpo_newencB/800 grpo_B_a2from700/1000_t10p95 grpo_newencB/900_t10p95 \
   grpo_newencB/800_rep100_t10p95 grpo_B_g3/800 grpo_B_a2from700/1000 grpo_newencB/1100_rep100 \
   grpo_B_nobleu/700 grpo_B_g3/700"
python label_guided_merge.py --draws $D --mode repair --min_support 3 --backbone 6 --max_add 5 --max_add_impression -1 --out ../results/GBF5I0.csv
python repair_verify_loop.py --merged ../results/GBF5I0.csv --draws $D --min_support 3 --iters 2 --max_add 2 --out ../results/GBF5I0_v.csv
```
Draw tags: `_t10p95` = temperature 1.0 / top-p 0.95, `_rep100` = repetition penalty 1.0;
default sampler is T 0.7 / top-p 0.9 / rep 1.05. The SLURM launchers that were actually used are
in `CTRG-3D/launchers/`.

### 6. Inference container

```bash
cd docker
bash make_weights.sh        # collects the files above into weights/ (or use the HF download)
bash build.sh && bash test.sh
docker run --gpus all -v <nii.gz dir>:/input -v <out>:/output -v $PWD/weights:/weights reportgen-pillar-ctrg3d
```
`predict.py` reproduces the full pipeline per volume: preprocessing, encoder, CLS and RET
templates, ten draws from the MLLM, RadBERT labelling, label vote, skeleton repair and the
verification loop. Output: `/output/predictions.json`.

## Citation

If you use this code or the released weights, please cite the repository (a paper will be
linked here when available):

```bibtex
@misc{kyung2026ctrg3d,
  title        = {CTRG-3D with GRPO post-training and label-vote ensemble: 2nd place in the VLM3D 2026 CT report generation challenge},
  author       = {Kyung, Sunggu},
  year         = {2026},
  howpublished = {\url{<GITHUB_REPO_URL>}},
  note         = {GitHub repository}
}
```

## Acknowledgements

This work builds on, and was inspired by, the following projects:

- **[Pillar-0](https://github.com/YalaLab/pillar-pretrain)** (Agrawal et al., 2025, *Pillar-0: A New Frontier for Radiology Foundation Models*) — the 3D CT encoder architecture, the public `Pillar0-ChestCT` weights we start from, and the [pillar-finetune](https://github.com/YalaLab/pillar-finetune) / [RaVE](https://github.com/YalaLab/rave) tooling our stages 1–4 are derived from.
- **[RadZero](https://github.com/ahnjonggwon/RadZero)** (Park et al., 2025, *RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Radiology with Zero-Shot Multi-Task Capability*) — the sentence-conditioned cross-attention alignment (VL-CABS) that our encoder pretraining objective extends with label-aware positives and negatives.
- **[PETRG-3D](https://github.com/Jorwnpay/PETRG-3D)** (arXiv 2511.20145) — the 3D encoder → Perceiver → LLM report-generation framework and training code that stage 5 is built on.
- **[TranSQ](https://github.com/zjukg/TranSQ)** (Kong et al., MICCAI 2022, *TranSQ: Transformer-based Semantic Query for Medical Report Generation*) — the query-based sentence retrieval formulation behind our retrieval head and `<ret_template>`.

We also thank the authors of [Qwen3](https://github.com/QwenLM/Qwen3) and [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding), [TRL](https://github.com/huggingface/trl) (GRPO), [PEFT](https://github.com/huggingface/peft), and the VLM3D Challenge organisers for the CT-RATE dataset and the RadBERT-based evaluation kit.

Fine-tuned weights are trained on CT-RATE (CC BY-NC-SA 4.0) and are released for non-commercial research use.
