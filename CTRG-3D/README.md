# CTRG-3D (stage 5: MLLM training, GRPO, draws, label-vote ensemble)

Training code for the report-generation MLLM: frozen LAPA encoder → Perceiver (128 visual
tokens) → Qwen3-8B + LoRA, prompted with the CLS and RET templates produced in stages 3–4.
Commands, data layout and the exact launcher chain used for the submission are in the
[root README](../README.md#5-mllm-sft--grpo--ensemble).

```
src/            model (Model/CTRG_3D.py), dataset (Dataset/), training entry (train.py), scorer (evaluate_reportgen.py)
configs/train/  SFT config chain: qwen3_joint_template.sh -> _new_encoder_template.sh -> _gtweight.sh
scripts/        vision cache, GT sentence weights, GRPO training and rewards, batched generation,
                sentence labelling, label-guided merge, repair / verify loop
launchers/      SLURM launchers in the order they were run (see root README)
storm/          helpers for running GRPO on a stand-alone GPU host
ds_configs/     DeepSpeed configs
data/sentence_labels/, GT_report_sentence_label/   GT-derived sentence spans and labels (small JSON)
```

The code is a fork of [PETRG-3D](https://github.com/Jorwnpay/PETRG-3D) adapted to CT-RATE chest CT
(single-modality CT input, template prompting, finding-weighted SFT, GRPO post-training). The
PET/CT-specific parts of the upstream repository are not included.
