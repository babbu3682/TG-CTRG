"""CTRG-3D main model definition."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# from .my_embedding_layer import MyEmbedding
from .pillar_embedding_layer import MyEmbedding


def weighted_causal_lm_loss(logits: torch.Tensor,
                            labels: torch.Tensor,
                            token_weight: torch.Tensor) -> torch.Tensor:
    """Causal-LM cross-entropy with a per-token weight, as a weighted mean.

    ``logits`` ``[B, L, V]``, ``labels`` and ``token_weight`` ``[B, L]``.

    The denominator is the sum of weights, not the token count, so the result
    is a *normalised* weighted mean: raising the weight changes only the ratio
    between finding tokens and boilerplate, never the absolute scale of the
    loss, and therefore never interacts with the learning rate. All-ones
    weights reproduce transformers' ``ForCausalLMLoss`` exactly, which
    scripts/audit_token_weight.py asserts numerically.

    Shifting is done by padding labels/weights on the right rather than by
    slicing ``logits[:, :-1, :]``. A slice is non-contiguous, so flattening it
    would materialise a second ``[B, L, V]`` copy in bf16 before ``.float()``
    makes a third -- 6.5 GB of it at per_device_train_batch_size=8 /
    max_seq=2688 / vocab 151,936. Padding the labels instead keeps
    ``logits.view`` a free view, which is also what ForCausalLMLoss does, so
    this path costs the same memory as the unweighted one it replaces.
    """
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    shift_weight = F.pad(token_weight, (0, 1), value=0.0)[..., 1:]

    flat_labels = shift_labels.reshape(-1)
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        flat_labels, reduction="none", ignore_index=-100,
    )
    keep = (flat_labels != -100).to(per_token.dtype)
    w = shift_weight.reshape(-1).to(per_token.dtype) * keep
    return (per_token * w).sum() / w.sum().clamp_min(1.0)


class CTRG3D(nn.Module):
    """CTRG3D-3D model.

    Wraps:
      * an LLM (loaded via ``AutoModelForCausalLM``) with a LoRA adapter
        attached through PEFT,
      * a shared 3D ViT visual encoder followed by two
        ``PerceiverResampler + FC`` heads -- one for CT --
        implemented in :class:`Model.my_embedding_layer.MyEmbedding`.
    """

    def __init__(
        self,
        text_tokenizer_path: str,
        lang_model_path: str,
        pretrained_visual_encoder: Optional[str],
        pretrained_adapter: Optional[str] = None,
        max_img_size: int = 1,
        image_num: int = 128,
        vis_dim: int = 768,
        use_fast: bool = False,
        enable_thinking: bool = False,
        finetune_ct_feature_extractor: bool = False,
        training_stage: str = "joint",
        model_config_yaml: str = None,
        weight_eval_loss: bool = True,
    ):
        super().__init__()
        self.weight_eval_loss = bool(weight_eval_loss)
        self.model_config_yaml = model_config_yaml
        self.enable_thinking = enable_thinking
        self.finetune_ct_feature_extractor = finetune_ct_feature_extractor
        self.training_stage = training_stage

        self.lang_model = AutoModelForCausalLM.from_pretrained(lang_model_path, trust_remote_code=True,)

        model_type = str(getattr(self.lang_model.config, "model_type", "")).lower()
        # GLM-family tokenizers require the fast implementation.
        self.use_fast = True if "glm" in model_type else bool(use_fast)

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, use_fast=self.use_fast, trust_remote_code=True,)

        # Step 1. Record the size of the LLM's native vocabulary before adding
        # any placeholder tokens so that we can correctly concatenate embeddings
        # later on.
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": []})
        # The resize is load-bearing, not cosmetic: Qwen3-8B ships 151,936
        # embedding rows for a 151,669-token vocabulary, and MyEmbedding appends
        # the wrapper/visual rows directly after the LLM's, so without shrinking
        # to len(tokenizer) every placeholder id would address an unused LLM
        # vocabulary row instead of a visual feature.
        self.lang_model.resize_token_embeddings(len(self.text_tokenizer))
        self.hidden_dim = self.lang_model.get_input_embeddings().embedding_dim
        self.voc_size = self.lang_model.get_input_embeddings().num_embeddings
        # Id of the first placeholder token = vocabulary size before Step 2.
        self.core_vocab_size = len(self.text_tokenizer)

        # Step 2. Add placeholder tokens for the modality wrappers and the visual
        # feature slots injected by :class:`MyEmbedding`.
        #
        # This order MUST match Dataset.ctrate_dataset_train.CTRATEDataset_Train's
        # special_token list exactly -- the dataset's tokenizer produces the
        # input_ids fed into this model's embedding table (see
        # MyEmbedding.forward in pillar_embedding_layer.py), so a mismatch here
        # silently misaligns token ids with embedding rows.
        special_token = {
            "additional_special_tokens": [
                "<ct>", "</ct>",
                "<cls_template>", "</cls_template>",
                "<ret_template>", "</ret_template>",
            ]
        }

        self.image_padding_tokens = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = f"<image_ct{i * image_num + j}>"
                padding += tok
                special_token["additional_special_tokens"].append(tok)
            self.image_padding_tokens.append(padding)

        self.text_tokenizer.add_special_tokens(special_token)

        # The dataset builds its own tokenizer from the same path with the same
        # list, so the two must end up with identical vocabulary sizes; if
        # image_num here and the dataset's image_num ever diverge, the visual
        # slots shift and the model reads features from the wrong positions.
        expected_vocab = self.core_vocab_size + 6 + max_img_size * image_num
        if len(self.text_tokenizer) != expected_vocab:
            raise ValueError(
                f"tokenizer has {len(self.text_tokenizer)} tokens after adding "
                f"placeholders, expected {expected_vocab} "
                f"(core {self.core_vocab_size} + 6 wrappers + "
                f"{max_img_size}x{image_num} image slots)."
            )

        # Step 3. Wrap the LLM with a LoRA adapter.
        lora_kwargs = dict(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,)
        if "glm" in model_type:
            # PEFT cannot auto-map the LoRA target modules for GLM variants.
            target_modules = ["query_key_value", "q_proj", "k_proj", "v_proj", "o_proj", "dense_h_to_4h", "dense_4h_to_h"]
            peft_config = LoraConfig(**lora_kwargs, target_modules=target_modules)
        else:
            peft_config = LoraConfig(**lora_kwargs)
        self.lang_model = get_peft_model(self.lang_model, peft_config)
        self.lang_model.print_trainable_parameters()

        try:
            self.lang_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()

        # Step 4. Build the multimodal embedding layer. (여기 수정.)
        self.embedding_layer = MyEmbedding(
            pretrained_visual_encoder=pretrained_visual_encoder,
            pretrained_adapter=pretrained_adapter,
            embedding_dim=self.hidden_dim,
            num_embeddings=self.voc_size,
            perceiver_num=image_num,
            core_vocab_size=self.core_vocab_size,
            vis_dim=vis_dim,
            finetune_ct_feature_extractor=self.finetune_ct_feature_extractor,
            model_config_yaml=self.model_config_yaml,
        )
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.loss_function = nn.CrossEntropyLoss()

        self._setup_trainable_params()

    # ------------------------------------------------------------------
    # Trainable-parameter schedules for the three supported training modes.
    # ------------------------------------------------------------------
    def _setup_trainable_params(self):
        if self.training_stage == "stage1":
            print("=" * 50)
            print("STAGE 1: Visual-Language Alignment")
            print("  Trainable: Vision Encoder + Perceiver + FC")
            print("  Frozen: LLM (including LoRA)")
            print("=" * 50)
            for param in self.lang_model.parameters():
                param.requires_grad = False
            for param in self.embedding_layer.vision_encoder.parameters():
                param.requires_grad = True
            for param in self.embedding_layer.perceiver.parameters():
                param.requires_grad = True
            for param in self.embedding_layer.fc.parameters():
                param.requires_grad = True

        elif self.training_stage == "stage2":
            print("=" * 50)
            print("STAGE 2: Instruction Tuning")
            print("  Trainable: Perceiver + FC + LLM (LoRA)")
            print("  Frozen: Vision Encoder")
            print("=" * 50)
            for param in self.embedding_layer.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.embedding_layer.perceiver.parameters():
                param.requires_grad = True
            for param in self.embedding_layer.fc.parameters():
                param.requires_grad = True
            for name, param in self.lang_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

        else:  # "joint"
            print("=" * 50)
            print("JOINT TRAINING: All modules train together")
            print(" (CT vision encoder follows `finetune_ct_feature_extractor`)")
            print("=" * 50)

        self._print_trainable_params()

    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"\nTrainable params: {trainable:,} || "
            f"Total params: {total:,} || "
            f"Trainable%: {100 * trainable / total:.2f}%\n"
        )

    # ------------------------------------------------------------------
    # Forward / generation entry points.
    # ------------------------------------------------------------------
    def forward(
        self,
        lang_x: torch.Tensor,
        vision_x: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        token_weight: Optional[torch.Tensor] = None,
    ):
        """``token_weight`` ``[B, L]`` reweights the per-token cross-entropy.

        Without it the loss is the language model's own: a uniform mean over
        every unmasked target token. That objective does not match how the
        report is scored. CRG and the classification metrics are read off
        RadBERT's 18 labels; only about a quarter of a CT-RATE report's
        sentences carry any of them and the rest is boilerplate the model
        already predicts easily -- so a model can drive eval_loss down by
        perfecting "Trachea and both main bronchi are open" while dropping a
        finding, and under CRG's r ~ 2.13 that omission costs ~4.26x what a
        wrong assertion does. Weighting the label-bearing tokens moves the
        objective onto the sentences the score actually reads.

        See :func:`weighted_causal_lm_loss` for the arithmetic and why the
        normalisation is what it is.
        """
        input_embedding = self.embedding_layer(vision_x, lang_x)
        model_dtype = next(self.lang_model.parameters()).dtype
        input_embedding = input_embedding.to(model_dtype)

        if token_weight is None or (not self.training and not self.weight_eval_loss):
            output = self.lang_model(inputs_embeds=input_embedding, attention_mask=attention_mask, labels=labels,)
            return dict(loss=output["loss"], logits=output["logits"])

        # labels=None: HF would otherwise compute the uniform loss being replaced.
        output = self.lang_model(inputs_embeds=input_embedding, attention_mask=attention_mask)
        logits = output["logits"]
        loss = weighted_causal_lm_loss(logits, labels, token_weight)
        return dict(loss=loss, logits=logits)

    def generate(self, lang_x: torch.Tensor, vision_x: Dict[str, torch.Tensor]):
        with torch.no_grad():
            input_embedding = self.embedding_layer(vision_x, lang_x)
            if self.enable_thinking:
                # Qwen3 "thinking" preset (see the Qwen3 HuggingFace model card).
                generation = self.lang_model.generate(
                    inputs_embeds=input_embedding,
                    do_sample=True,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    temperature=0.6,
                    repetition_penalty=1.05,
                    max_new_tokens=2048,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                )
            else:
                generation = self.lang_model.generate(
                    inputs_embeds=input_embedding,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.05,
                    max_new_tokens=2048, # 1024
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                )
            report = self.text_tokenizer.batch_decode(generation, skip_special_tokens=True)
            return report


# Backwards-compatible alias: previous versions of the codebase named the class
# ``Reg2RG`` (after the upstream repository it was forked from). Checkpoints only
# store parameter names, so the alias is kept only for any user code that still
# imports the old class name.
Reg2RG = CTRG3D
