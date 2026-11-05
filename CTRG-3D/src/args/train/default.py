"""Default training argument schema for PETRG-3D.

All paths are left as ``None`` / placeholders by default; concrete values
should be supplied through the shell configuration scripts under ``configs/``
or on the command line.
"""

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path or HF repo of the backbone LLM."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path or HF repo of the tokenizer (often the same as lang_encoder_path)."},
    )
    pretrained_visual_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to RadFM's 3D ViT checkpoint (RadFM_vit3d.pth)."},
    )
    pretrained_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "(Unused in the released config; kept for backwards compatibility.)"},
    )
    pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a previous PETRG-3D checkpoint to continue training from."},
    )
    finetune_ct_feature_extractor: bool = field(
        default=False,
        metadata={"help": "Whether to finetune the CT feature extractor (3D ViT) during training."},
    )
    training_stage: str = field(
        default="joint",
        metadata={"help": "Training stage: 'joint' (all modules), 'stage1' (vision+adapter), or 'stage2' (adapter+LLM)."},
    )

    # ADD
    model_config_yaml: str = field(
        default=None,
        metadata={"help": "Path to the file."},
    )
    vis_dim: int = field(
        default=768,
        metadata={"help": "Path to the file."},
    )
    num_visual_tokens: int = field(
        default=128,
        metadata={
            "help": "Number of visual tokens per volume: the PerceiverResampler's "
                    "latent count and, equivalently, the number of <image_ct*> "
                    "placeholder tokens in the prompt. One value drives both the "
                    "model and the dataset -- they must never diverge. Raising it "
                    "lowers the compression of the encoder's 65,536 output tokens "
                    "but costs perceiver memory and max_seq (each visual token is "
                    "one LLM position)."
        },
    )


@dataclass
class DataArguments:
    # Main data paths for PET/CT.
    petct_image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing paired {id}_0000.nii.gz (CT) and {id}_0001.nii.gz (PET) volumes."},
    )
    petct_report_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing Chinese JSON reports named {id}.json."},
    )

    # Image preprocessing.
    z_chunk_size: int = field(default=64)
    z_chunk_stride: int = field(default=32)
    xy_size: int = field(default=256)
    pet_clip_percentile: float = field(default=99.5)
    pet_apply_log: bool = field(default=True)

    # Text / LLM handling.
    max_seq: int = field(default=2048)
    use_fast: bool = field(default=False, metadata={"help": "Use the fast tokenizer implementation."})
    use_template: bool = field(default=False, metadata={"help": "Wrap the answer into the Chinese report template."})
    template_path: Optional[str] = field(default=None, metadata={"help": "Path to the JSON template file."})
    use_cls_template: bool = field(default=False, metadata={"help": "Prepend the classification model's key-phrase predictions to the prompt."})
    use_ret_template: bool = field(default=False, metadata={"help": "Prepend the retrieval model's key-phrase predictions to the prompt."})
    cls_template_path: Optional[str] = field(default=None, metadata={"help": "Path to the classification model's JSON output (e.g. abn_reports.json)."})
    ret_template_path: Optional[str] = field(default=None, metadata={"help": "Path to the retrieval model's JSON output (e.g. ret_reports.json)."})
    enable_thinking: bool = field(default=False, metadata={"help": "Enable Qwen3-style 'thinking' generation."})
    enable_augmentation: bool = field(default=False, metadata={"help": "Enable on-the-fly volume augmentation."})

    # ADD
    train_json_path: Optional[str] = field(default=None, metadata={"help": "Path to the file."})
    train_rve_manifest_path: Optional[str] = field(default=None, metadata={"help": "Path to the file."})

    # Validation split (optional -- leave valid_json_path unset to train without
    # an eval loop, matching the previous behavior).
    valid_json_path: Optional[str] = field(default=None, metadata={"help": "Path to the validation-split JSONL manifest. If unset, no eval_dataset is built."})
    valid_rve_manifest_path: Optional[str] = field(default=None, metadata={"help": "Path to the validation-split RVE cache manifest CSV."})
    valid_cls_template_path: Optional[str] = field(default=None, metadata={"help": "Path to the classification model's JSON output for the validation split."})
    valid_ret_template_path: Optional[str] = field(default=None, metadata={"help": "Path to the retrieval model's JSON output for the validation split."})

    # Template regularization. Declared here rather than left to
    # CTRATEDataset_Train's constructor defaults so that the values land in the
    # logged training arguments: an ablation is only interpretable if every arm
    # provably shares them (see the dropout block in ctrate_dataset_train.py).
    cls_dropout_prob: float = field(default=0.30, metadata={"help": "Per-sentence dropout probability inside the <cls_template> block (train split only)."})
    ret_dropout_prob: float = field(default=0.30, metadata={"help": "Per-sentence dropout probability inside the <ret_template> block (train split only)."})
    template_full_drop_prob: float = field(default=0.05, metadata={"help": "Probability of dropping a whole template block, so the model also trains on the aux-free prompt and does not depend on a block being present."})

    # Token-weighted loss. Inert at these defaults: with weight 1.0 the dataset
    # emits no token_weight at all and CTRG3D uses the language model's own
    # uniform cross-entropy, so every existing config is bit-identical.
    sentence_labels_path: Optional[str] = field(default=None, metadata={"help": "scripts/label_report_sentences.py output for the train split (character spans of report_metadata's sentences + whether the scorer's RadBERT fires on each)."})
    valid_sentence_labels_path: Optional[str] = field(default=None, metadata={"help": "Same, for the validation split. Required whenever weight_eval_loss is True and label_token_weight != 1.0."})
    label_token_weight: float = field(default=1.0, metadata={"help": "Cross-entropy weight on target tokens inside sentences the scorer's RadBERT labels; every other target token stays at 1.0. The loss is a normalised weighted mean, so this changes only the ratio, never the scale. 1.0 disables weighting entirely."})
    weight_eval_loss: bool = field(default=True, metadata={"help": "Apply the same weighting to eval_loss. True keeps load_best_model_at_end and early stopping selecting on the objective actually being optimised; False reports the plain uniform loss instead, which stays numerically comparable with unweighted runs."})

    # Small-scale ablation support.
    train_subsample_fraction: float = field(default=1.0, metadata={"help": "Fraction of the train split to keep (0 < f <= 1). Deterministic in subsample_seed alone, so all ablation arms see the identical subset."})
    valid_subsample_fraction: float = field(default=1.0, metadata={"help": "Fraction of the validation split to keep (0 < f <= 1). Same determinism guarantee as train_subsample_fraction."})
    subsample_seed: int = field(default=42, metadata={"help": "Seed for both subsample draws. Changing it changes the subset, so keep it fixed across the arms being compared."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # NOTE: named eval_strategy, not evaluation_strategy -- the installed
    # transformers (4.57.0) dropped evaluation_strategy entirely in favor of
    # eval_strategy, with no backward-compat alias. Redeclaring it under the
    # old name here would silently no-op: it'd set an unused field on this
    # dataclass while the base class's actual self.eval_strategy stayed at
    # its own default, breaking load_best_model_at_end's
    # save/eval-strategy-match check with a confusing error.
    eval_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to adopt during training: 'no', 'steps' or 'epoch'."},
    )
    output_dir: Optional[str] = field(
        default="./outputs/petrg_3d_run",
        metadata={"help": "Base directory where checkpoints and logs are written."},
    )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    pin_memory: bool = field(default=True)
    # Early stopping. transformers keeps this in EarlyStoppingCallback rather
    # than in TrainingArguments, so it needs a field here for HfArgumentParser
    # to accept the flag; train.py turns it into the callback. Counted in
    # evaluation calls, which with eval_strategy="epoch" means epochs without
    # eval_loss improvement. 0 disables it.
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Stop after this many evaluations without metric_for_best_model improving. 0 disables early stopping."},
    )
