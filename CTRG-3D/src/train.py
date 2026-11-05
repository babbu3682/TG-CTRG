import os

import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from transformers import Trainer

# from Model.PETRG_3D import PETRG3D
from Model.CTRG_3D import CTRG3D
# from Dataset.petct_dataset_train import PETCTDataset_Train
from Dataset.ctrate_dataset_train import CTRATEDataset_Train
from args.train.default import ModelArguments, DataArguments, TrainingArguments


@dataclass
class DataCollator(object):
    """Collate a batch of CT + text samples produced by CTRATEDataset_Train."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        lang_xs, vision_xs, attention_masks, labels = tuple(
            [instance[key] for instance in instances]
            for key in ('lang_x', 'vision_x', 'attention_mask', 'label')
        )

        lang_xs = torch.cat([x.unsqueeze(0) for x in lang_xs], dim=0)
        attention_masks = torch.cat([x.unsqueeze(0) for x in attention_masks], dim=0)
        labels = torch.cat([x.unsqueeze(0) for x in labels], dim=0)

        ct_images = torch.cat([v['ct_image'].unsqueeze(0) for v in vision_xs], dim=0)
        # pet_images = torch.cat([v['pet_image'].unsqueeze(0) for v in vision_xs], dim=0)
        ct_images = ct_images.bfloat16()  # add

        batch = dict(
            lang_x=lang_xs,
            vision_x={'ct_image': ct_images},
            attention_mask=attention_masks,
            labels=labels,
        )
        # Only present when the dataset was given sentence labels *and* a weight
        # other than 1.0; without it CTRG3D falls back to the language model's
        # own uniform cross-entropy. The dataset pads every sample to max_seq,
        # so the stack needs no padding of its own -- but padding with 1.0 (not
        # 0.0) is the rule if that ever changes, because a 0 would shrink the
        # weighted mean's denominator and silently rescale the loss.
        if 'token_weight' in instances[0]:
            batch['token_weight'] = torch.cat(
                [x['token_weight'].unsqueeze(0) for x in instances], dim=0)
        return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
def main():
    set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("================= Arguments ================")
    print(f"data_args: {data_args}")
    print(f"model_args: {model_args}")
    print(f"training_args: {training_args}")
    print("==========================================")

    print("Setup Data")
    Train_dataset = CTRATEDataset_Train(
        tokenizer_path=model_args.tokenizer_path,
        image_folder=data_args.petct_image_folder,
        report_folder=data_args.petct_report_folder,
        z_chunk_size=data_args.z_chunk_size,
        z_chunk_stride=data_args.z_chunk_stride,
        xy_size=(data_args.xy_size, data_args.xy_size),
        max_seq=data_args.max_seq,
        use_fast=data_args.use_fast,
        use_cls_template=data_args.use_cls_template,
        use_ret_template=data_args.use_ret_template,
        cls_template_path=data_args.cls_template_path,
        ret_template_path=data_args.ret_template_path,
        enable_thinking=data_args.enable_thinking,
        json_path=data_args.train_json_path,
        rve_manifest_path=data_args.train_rve_manifest_path,
        image_num=model_args.num_visual_tokens,
        cls_dropout_prob=data_args.cls_dropout_prob,
        ret_dropout_prob=data_args.ret_dropout_prob,
        template_full_drop_prob=data_args.template_full_drop_prob,
        subsample_fraction=data_args.train_subsample_fraction,
        subsample_seed=data_args.subsample_seed,
        sentence_labels_path=data_args.sentence_labels_path,
        label_token_weight=data_args.label_token_weight,
    )

    Valid_dataset = None
    if data_args.valid_json_path:
        print("Setup Validation Data")
        Valid_dataset = CTRATEDataset_Train(
            tokenizer_path=model_args.tokenizer_path,
            image_folder=data_args.petct_image_folder,
            report_folder=data_args.petct_report_folder,
            z_chunk_size=data_args.z_chunk_size,
            z_chunk_stride=data_args.z_chunk_stride,
            xy_size=(data_args.xy_size, data_args.xy_size),
            max_seq=data_args.max_seq,
            use_fast=data_args.use_fast,
            use_cls_template=data_args.use_cls_template,
            use_ret_template=data_args.use_ret_template,
            cls_template_path=data_args.valid_cls_template_path,
            ret_template_path=data_args.valid_ret_template_path,
            enable_thinking=data_args.enable_thinking,
            json_path=data_args.valid_json_path,
            rve_manifest_path=data_args.valid_rve_manifest_path,
            image_num=model_args.num_visual_tokens,
            is_train=False,
            # No dropout probabilities: is_train=False skips the dropout block
            # entirely, so eval always sees the complete templates.
            subsample_fraction=data_args.valid_subsample_fraction,
            subsample_seed=data_args.subsample_seed,
            # The eval split is weighted too, so that eval_loss -- which drives
            # load_best_model_at_end and EarlyStoppingCallback -- measures the
            # objective actually being optimised. See weight_eval_loss below
            # for the knob that turns this back into the plain uniform loss.
            sentence_labels_path=data_args.valid_sentence_labels_path,
            label_token_weight=data_args.label_token_weight,
        )

    print("Setup Model")
    model = CTRG3D(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        pretrained_adapter=model_args.pretrained_adapter,
        use_fast=data_args.use_fast,
        enable_thinking=data_args.enable_thinking,
        finetune_ct_feature_extractor=model_args.finetune_ct_feature_extractor,
        training_stage=model_args.training_stage,
        model_config_yaml=model_args.model_config_yaml,
        vis_dim=model_args.vis_dim,
        image_num=model_args.num_visual_tokens,
        weight_eval_loss=data_args.weight_eval_loss,
    )
    # add
    model = model.bfloat16()

    # Load pretrained model weights if specified
    if model_args.pretrained_model_path is not None:
        print(f"Loading pretrained model weights from: {model_args.pretrained_model_path}")
        ckpt = torch.load(model_args.pretrained_model_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)
        print("Pretrained model weights loaded successfully!")
        model = model.bfloat16()  # add

    # EarlyStoppingCallback fires on each evaluation, so with
    # eval_strategy="epoch" the patience is measured in epochs. It needs the
    # eval loop and metric_for_best_model to exist; failing loudly here beats
    # transformers' own mid-run assertion after hours of training.
    callbacks = []
    if training_args.early_stopping_patience > 0:
        if Valid_dataset is None or training_args.eval_strategy == "no":
            raise ValueError(
                "early_stopping_patience > 0 requires a validation split "
                "(valid_json_path) and eval_strategy != 'no'."
            )
        callbacks.append(transformers.EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
        ))

    trainer = Trainer(model=model,
                      train_dataset=Train_dataset,
                      eval_dataset=Valid_dataset,
                      args=training_args,
                      data_collator=DataCollator(),
                      callbacks=callbacks or None,
                      )

    # TrainingArguments carries resume_from_checkpoint, but Trainer.train() does
    # NOT read it on its own -- it has to be handed over explicitly, otherwise
    # passing --resume_from_checkpoint silently starts from step 0 and overwrites
    # the epoch checkpoints already on disk.
    #
    # Accepts a checkpoint-* directory, or "True"/"1" to let the Trainer pick the
    # newest checkpoint under output_dir itself.
    resume = training_args.resume_from_checkpoint
    if isinstance(resume, str) and resume.strip().lower() in {"true", "1", "yes"}:
        resume = True
    elif isinstance(resume, str) and not resume.strip():
        resume = None
    if resume:
        print(f"Resuming training from: {resume}")

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
      
if __name__ == "__main__":
    main()