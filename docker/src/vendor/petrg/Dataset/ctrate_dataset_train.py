import os
import glob
import json
import math
import inspect
from typing import Dict, List, Tuple, Any, Optional

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset
import monai.transforms as transforms
from transformers import AutoConfig, AutoTokenizer
import pandas as pd

# Import RVE for loading tar files
try:
    import rve
    HAS_RVE = True
except ImportError:
    HAS_RVE = False
    print("Warning: rad-vision-engine (rve) not installed. RVE loading will be disabled.")


def load_rve_volume(filepath: str,
                   target_slices: Optional[int] = None,
                   target_h: Optional[int] = None,
                   target_w: Optional[int] = None,
                   pad_value: float = -1024.0,
                   crop_mode: str = 'center') -> torch.Tensor:
    """
    Load a 3D volume from an RVE tar file with optional cropping/padding to target dimensions.

    Args:
        filepath: Path to the .tar file from RVE
        target_slices: Target number of slices (D dimension). If provided, will pad or crop.
        target_h: Target height (H dimension). If provided, will pad or crop.
        target_w: Target width (W dimension). If provided, will pad or crop.
        pad_value: Value to use for padding (default: -1024 for CT)
        crop_mode: How to crop when volume is larger than target ('center' or 'random')

    Returns:
        Volume tensor with shape (1, D, H, W)
    """
    try:
        # Load the tar file using RVE with CPU decode (26x faster than GPU decode)
        # RVE returns a torch tensor already
        volume = rve.load_sample(filepath, use_hardware_acceleration=False)

        # RVE should return shape (D, H, W), add channel dimension if needed
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)  # Add channel dimension: (D, H, W) -> (1, D, H, W)
        elif volume.dim() != 4:
            raise ValueError(f"Unexpected volume shape from RVE: {volume.shape}")

        # Pad Z dimension if needed
        if target_slices is not None and volume.shape[1] < target_slices:
            z_diff = target_slices - volume.shape[1]
            z_pad_before = z_diff // 2
            z_pad_after = z_diff - z_pad_before
            # Pad only the Z dimension (dimension 1 after channel)
            padding = (0, 0, 0, 0, z_pad_before, z_pad_after)  # (W, H, D) padding in reverse order
            volume = torch.nn.functional.pad(volume, padding, mode='constant', value=pad_value)
        elif target_slices is not None and volume.shape[1] > target_slices:
            # Crop D dimension
            if crop_mode == 'random':
                # Random crop for training
                start_d = torch.randint(0, volume.shape[1] - target_slices + 1, (1,)).item()
                volume = volume[:, start_d:start_d + target_slices]
            else:
                # Center crop for validation/inference
                start_d = (volume.shape[1] - target_slices) // 2
                volume = volume[:, start_d:start_d + target_slices]

        # Handle H dimension (height)
        if target_h is not None:
            current_h = volume.shape[2]
            if current_h < target_h:
                # Pad H dimension
                h_diff = target_h - current_h
                h_pad_before = h_diff // 2
                h_pad_after = h_diff - h_pad_before
                padding = (0, 0, h_pad_before, h_pad_after, 0, 0)  # (W, H, D) padding in reverse order
                volume = torch.nn.functional.pad(volume, padding, mode='constant', value=pad_value)
            elif current_h > target_h:
                # Crop H dimension
                start_h = (current_h - target_h) // 2
                volume = volume[:, :, start_h:start_h + target_h, :]

        # Handle W dimension (width)
        if target_w is not None:
            current_w = volume.shape[3]
            if current_w < target_w:
                # Pad W dimension
                w_diff = target_w - current_w
                w_pad_before = w_diff // 2
                w_pad_after = w_diff - w_pad_before
                padding = (w_pad_before, w_pad_after, 0, 0, 0, 0)  # (W, H, D) padding in reverse order
                volume = torch.nn.functional.pad(volume, padding, mode='constant', value=pad_value)
            elif current_w > target_w:
                # Crop W dimension
                start_w = (current_w - target_w) // 2
                volume = volume[:, :, :, start_w:start_w + target_w]

        return volume
    except Exception as e:
        print(f"Failed to load RVE volume from {filepath}: {e}")
        raise


#: Root the RVE manifests' relative ``image_cache_path`` entries are written
#: against (pillar-pretrain's own working directory).
RVE_ROOT = "/data/drdcad/sunggu/0.Challenge/pillar-pretrain/"


def _resolve_rve_path(image_cache_path: Optional[str], sample_name: str) -> str:
    """Turn a manifest ``image_cache_path`` into an absolute cache path.

    The manifests are not consistent with each other: the 1.25mm video caches
    store paths relative to ``RVE_ROOT`` ("data/ctrate/...") while the newer
    lz4 caches (ctrate_cache_1.25mm_lz4) store absolute ones. Blindly
    prefixing, as this used to do, silently produces
    ``.../pillar-pretrain//data/...`` for the lz4 manifests and every sample
    fails to load, so branch on it here.
    """
    if image_cache_path is None or not isinstance(image_cache_path, str) or not image_cache_path.strip():
        raise KeyError(
            f"sample_name={sample_name!r} has no image_cache_path in the RVE "
            f"manifest -- check that the manifest matches the split json."
        )
    image_cache_path = image_cache_path.strip()
    if os.path.isabs(image_cache_path):
        return image_cache_path
    return os.path.join(RVE_ROOT, image_cache_path)


def _resolve_message_type(tokenizer_path: str, override: str = None) -> str:
    """Return the prompt-formatting style ('chatml' or 'human_design').

    ``human_design`` is the [INST]/[/INST]-style prompt used by Llama-2 Chinese
    Chat checkpoints; every other supported backbone uses the ChatML-style
    template emitted by ``tokenizer.apply_chat_template``. Callers may override
    the auto-detection by passing a concrete value.
    """
    if override is not None:
        return override
    try:
        model_type = str(getattr(AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True), "model_type", "")).lower()
    except Exception:
        model_type = ""
    if "llama" in model_type:
        return "human_design"
    return "chatml"


def threshold_ct(x):
    return x > -1000


#: System turn, shared by CTRATEDataset_Train and CTRATEDataset_Test so the two
#: can never drift apart. Deliberately role-only: the task, the constraints and
#: the output format all live in the user turn (see build_instruction), which
#: keeps each instruction in exactly one place.
SYSTEM_PROMPT = (
    "You are an expert thoracic radiologist. You read chest CT studies and write "
    "the Findings and Impression sections of the report."
)

#: What each auxiliary source actually is. Rendered on the opening tag line so
#: the description travels with the block (a block can be dropped by
#: sentence/template dropout at train time, or be absent because the model had
#: no usable prediction). The two differ in failure mode, which is why they are
#: named rather than lumped together as "AI predictions": cls is a per-label
#: presence call, whereas ret consists of sentences copied from *other*
#: patients' reports and therefore carries invented-looking specifics
#: (segments, laterality, sizes) that must be checked against the images.
AUX_SOURCE_LABELS = {
    "cls": "per-abnormality presence calls from a classifier",
    "ret": "sentences retrieved from reports of visually similar cases",
}


def build_instruction(cls_template: Optional[str] = None, ret_template: Optional[str] = None) -> str:
    """
    Builds the user-turn text given to the LLM, optionally including auxiliary
    predictions from a classification model (cls_template) and/or a retrieval
    model (ret_template).

    Layout: task line -> auxiliary evidence -> rules -> output format. The
    request comes first so the model knows what it is reading the auxiliary
    blocks for, and the format spec comes last, closest to the generation.

    Design notes:
    - Both templates are AI predictions and may be individually wrong, and
      may also disagree with each other. Rather than trying to programmatically
      detect agreement/conflict between the two (infeasible given free-text
      key phrases with unbounded phrasing), we surface both sources verbatim,
      tagged by origin, and delegate the reconciliation judgment to the LLM
      itself via explicit instructions.
    - The CT images remain the primary evidence source. That instruction is
      given once, not restated in four different wordings: repeating it does
      not make it stronger, it just dilutes the rest of the prompt.
    - The auxiliary-specific rules are emitted only when at least one block is
      actually present, so a prompt never refers to evidence it does not show.
    - This function is used identically at both train and inference time, so
      the model is never trained on a prompt format it won't see at inference.
    """

    template_blocks = []

    if cls_template:
        template_blocks.append(
            f"<cls_template> ({AUX_SOURCE_LABELS['cls']})\n"
            f"{cls_template.strip()}\n"
            f"</cls_template>"
        )

    if ret_template:
        template_blocks.append(
            f"<ret_template> ({AUX_SOURCE_LABELS['ret']})\n"
            f"{ret_template.strip()}\n"
            f"</ret_template>"
        )

    if template_blocks:
        aux_section = (
            "Auxiliary predictions from AI models are given below for reference "
            "only. They are unverified and may be wrong or incomplete. A block "
            "may be absent, which carries no meaning -- it is not a normal "
            "finding.\n\n"
            + "\n\n".join(template_blocks)
            + "\n\n"
        )
        rules = (
            "Rules:\n"
            "- The images are the evidence. Verify every finding against them, and "
            "where the predictions disagree, decide from the images.\n"
            "- Do not state any finding, location, laterality, measurement, or "
            "diagnosis the images do not support, and do report relevant findings "
            "the predictions omit.\n"
            "- Use cautious language where the images are equivocal, and never "
            "refer to the auxiliary predictions.\n\n"
        )
    else:
        aux_section = ""
        rules = (
            "Rules:\n"
            "- Report only what the images support: no finding, location, "
            "laterality, measurement, or diagnosis beyond the imaging evidence.\n"
            "- Use cautious language where the images are equivocal.\n\n"
        )

    instruction = (
        "Write the radiology report for this chest CT.\n\n"
        f"{aux_section}"
        f"{rules}"
        "Output exactly two lines, in English:\n"
        "Findings: detailed description of the imaging observations.\n"
        "Impression: concise clinical interpretation and conclusion."
    )

    return instruction


import re


def sentence_dropout(
    text: Optional[str],
    drop_prob: float = 0.15,
) -> Optional[str]:
    """
    Randomly drops sentences from a report template.

    Args:
        text:
            Input report template. Must be a plain string (or None) -- if the
            underlying template file stores structured JSON per sample
            (e.g. {"report": ..., "key_phrases": [...], "status": {...}}),
            flatten it to a single string (e.g. the "report" field, or a
            joined form of "key_phrases") before calling this function.
        drop_prob:
            Probability of dropping each sentence.

    Returns:
        Template after sentence dropout.
        At least one sentence is preserved.
    """
    if text is None:
        return None

    if not isinstance(text, str):
        raise TypeError(
            f"sentence_dropout expected a str or None, got {type(text).__name__}. "
            f"Flatten structured template records into a single string before "
            f"calling sentence_dropout()."
        )

    text = text.strip()

    if not text:
        return None

    if not 0.0 <= drop_prob < 1.0:
        raise ValueError("drop_prob must satisfy 0.0 <= drop_prob < 1.0")

    if drop_prob == 0.0:
        return text

    # Split by newline or English sentence boundary.
    # Decimal values such as 1.2 cm are not split.
    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[\r\n]+|(?<=[.!?])\s+(?=[A-Z])",
            text,
        )
        if sentence.strip()
    ]

    if len(sentences) <= 1:
        return text

    keep_mask = torch.rand(len(sentences)) >= drop_prob

    # Prevent the entire template from disappearing.
    if not keep_mask.any():
        keep_index = torch.randint(
            low=0,
            high=len(sentences),
            size=(1,),
        ).item()
        keep_mask[keep_index] = True

    kept_sentences = [
        sentence
        for sentence, keep in zip(sentences, keep_mask.tolist())
        if keep
    ]

    return "\n".join(kept_sentences)


def _lookup_template(template_dict: dict, sample_name: str, template_name: str) -> Optional[str]:
    """
    Looks up a per-sample template string from a loaded template dict.

    Distinguishes two failure modes instead of silently returning None for both:
      - sample_name not present as a key at all -> likely a data pipeline
        issue (e.g. manifest/template files out of sync); logs a warning.
      - value present but None/empty -> treated as "no prediction available
        for this sample", which is an expected, silent case.

    Also guards against templates stored as structured JSON (dict/list)
    instead of a flat string, which would otherwise silently corrupt
    sentence_dropout()'s sentence-splitting logic.
    """
    if sample_name not in template_dict:
        print(f"[warn] {template_name} template missing for sample_name={sample_name!r} "
              f"(key not found in template file; check that the template file "
              f"and dataset manifest are in sync)")
        return None

    value = template_dict[sample_name]

    if value is not None and not isinstance(value, str):
        raise TypeError(
            f"{template_name} template for sample_name={sample_name!r} is "
            f"{type(value).__name__}, expected str or None. sentence_dropout() "
            f"requires a flat string; pre-flatten structured JSON records "
            f"(e.g. 'report'/'key_phrases' fields) into a single string when "
            f"building the template file."
        )

    return value


def _load_key_phrase_template(
    path: str,
    template_name: str,
    key_phrase_fields: Tuple[str, ...] = ("pred_key_phrases", "report"),
    id_field: str = "accession",
) -> Dict[str, Optional[str]]:
    """
    Loads a cls/ret model output file into the ``{sample_name: template_string}``
    lookup consumed by ``_lookup_template()``.

    Two record layouts are supported, both a list of per-sample records:
      * ``{"accession": ..., "pred_key_phrases": [...]}`` -- the retrieval
        model's ``*_ret_reports.json``, and the older abnormality-classifier
        ``*_abn_reports.json``.
      * ``{"accession": ..., "report": [...], "status_<label>": ...}`` -- the
        threshold-based classifier report templates
        (``cls_report_{train,dev}.json``), whose ``report`` field is already a
        list of finished sentences ("No cardiomegaly is present."). The
        ``status_*`` fields are per-label positive/negative/uncertain calls and
        are intentionally ignored: only the sentences the generator decided to
        emit (its p90/n20 thresholds) go into the prompt.

    The field is picked per record from ``key_phrase_fields`` in order, so a
    single loader handles both without the caller having to know which file it
    was handed.

    Each record's phrase list is newline-joined into a single string so that
    ``sentence_dropout()`` can drop individual phrases independently. Phrases
    that are missing, "", or whitespace-only carry no information and are
    dropped; if a record has no usable phrases left, its template is stored as
    None so downstream lookups skip it (see ``_lookup_template`` and
    ``build_instruction``) instead of injecting an empty
    ``<..._template></..._template>`` block into the prompt.
    """
    records = json.load(open(path, "r"))
    if not isinstance(records, list):
        raise TypeError(
            f"{template_name} template file {path!r} must be a list of "
            f"per-sample records (each with {id_field!r} and one of "
            f"{list(key_phrase_fields)} keys), got {type(records).__name__}."
        )

    templates: Dict[str, Optional[str]] = {}
    used_fields = set()
    for record in records:
        sample_name = record.get(id_field)
        if sample_name is None:
            continue

        phrases = None
        for field in key_phrase_fields:
            if field in record:
                phrases = record[field]
                used_fields.add(field)
                break
        if isinstance(phrases, str):
            phrases = [phrases]
        phrases = phrases or []

        cleaned = [p.strip() for p in phrases if isinstance(p, str) and p.strip()]
        templates[sample_name] = "\n".join(cleaned) if cleaned else None

    if not used_fields:
        raise KeyError(
            f"{template_name} template file {path!r} has no {list(key_phrase_fields)} "
            f"field in any record; first record's keys: "
            f"{list(records[0].keys()) if records else '<empty file>'}"
        )

    non_empty = sum(1 for v in templates.values() if v)
    print(
        f"[info] {template_name} templates loaded from {path}: {len(templates)} samples "
        f"({non_empty} non-empty) via field(s) {sorted(used_fields)}"
    )

    return templates


class CTRATEDataset_Train(Dataset):
    def __init__(
        self,
        tokenizer_path: str,
        image_folder: str,
        report_folder: str,
        cls_template_path: str = None,
        ret_template_path: str = None,
        z_chunk_size: int = 64,
        z_chunk_stride: int = 32,
        xy_size: Tuple[int, int] = (256, 256),
        max_img_size: int = 1,
        image_num: int = 128,
        max_seq: int = 4096,
        use_fast: bool = False,
        use_cls_template: bool = False,
        use_ret_template: bool = False,
        enable_thinking: bool = False,
        message_type: str = None,
        json_path: str = None,
        rve_manifest_path: str = None,
        target_d: int = 256,
        target_hw: int = 256,
        pad_value: int = -1024,
        is_train: bool = True,
        cls_dropout_prob: float = 0.30,
        ret_dropout_prob: float = 0.30,
        template_full_drop_prob: float = 0.05,
        subsample_fraction: float = 1.0,
        subsample_seed: int = 42,
        sentence_labels_path: str = None,
        label_token_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.is_train = is_train
        self.cls_dropout_prob = cls_dropout_prob
        self.ret_dropout_prob = ret_dropout_prob
        self.template_full_drop_prob = template_full_drop_prob
        self.target_d = target_d
        self.target_hw = target_hw
        self.pad_value = pad_value

        self.use_cls_template = use_cls_template
        self.use_ret_template = use_ret_template
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast, trust_remote_code=True)
        # NOTE: len(tokenizer), not tokenizer.vocab_size. Backbones like Qwen3
        # ship extra pre-existing added tokens (<|im_end|>, <think>, ...) that
        # count toward len(tokenizer) but not vocab_size; label-masking below
        # must not treat those as "placeholder" ids or the model would never
        # get loss signal for its own eos/turn-end token. This also matches
        # Model.CTRG_3D.CTRG3D's self.voc_size, which is likewise captured
        # from len(tokenizer) before the wrapper/image tokens are added.
        self.core_vocab_size = len(self.tokenizer)

        # 특수 토큰: CT 감싸는 태그 및 플레이스홀더
        special_token = {"additional_special_tokens": ["<ct>", "</ct>", "<cls_template>", "</cls_template>", "<ret_template>", "</ret_template>"]}

        # CT 이미지의 특징 플레이스홀더 생성: <image_ct1>, <image_ct2>, ...
        self.image_padding_tokens: List[str] = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = "<image_ct" + str(i * image_num + j) + ">"
                padding += tok
                special_token["additional_special_tokens"].append(tok)  # 새로 추가된 토큰을 special_token 리스트에 저장
            self.image_padding_tokens.append(padding)
        '''
        NOTE:
        (1) 튜토리얼에서는 add_special_tokens로 새 토큰을 추가한 후 model.resize_token_embeddings(len(tokenizer))를 호출하여 모델의 embedding 크기를 업데이트해야 한다고 하지만, 현재 코드 어디에도 해당 호출이 없어 의문이 생겼습니다.
        → 이후 my_embedding에서 처리됩니다.
        (2) 또한 현재 코드에서 왜 special token을 추가하는가?
        → 특징(feature)은 어휘 사전의 토큰이 아니므로, 이후 사용을 위해 special_token에 추가해야 합니다.
        '''
        self.tokenizer.add_special_tokens(special_token)  # 특수 토큰을 토크나이저의 어휘 사전에 추가

        # ------------------------------------------------------------------
        # Token-weighted loss (see CTRG3D.forward and _token_weight below).
        #
        # `sentence_labels_path` is scripts/label_report_sentences.py's output:
        # for every sample_name, the character spans of report_metadata's
        # sentences and whether the VLM3D scorer's own RadBERT fires on each.
        # Both knobs are inert at their defaults -- without a path, or with
        # weight 1.0, no `token_weight` is emitted at all and the model falls
        # back to the language model's uniform cross-entropy, so every existing
        # config behaves exactly as before.
        # ------------------------------------------------------------------
        self.label_token_weight = float(label_token_weight)
        self.sentence_labels = None
        if sentence_labels_path and self.label_token_weight != 1.0:
            with open(sentence_labels_path, encoding='utf-8') as f:
                self.sentence_labels = json.load(f)
            print(f'[token weight] {len(self.sentence_labels)} volumes from '
                  f'{sentence_labels_path}, weight {self.label_token_weight}')

        # The configured tokenizer is the slow one (use_fast defaults to False
        # in configs/train/*.sh) and Python tokenizers cannot return
        # offset_mapping, so a fast twin is built for that one purpose -- same
        # path, same added special tokens, used only to locate characters.
        # _token_weight asserts the two agree token-for-token on every sample
        # it weights, and dies if they ever do not: silently misaligned weights
        # are the most expensive failure mode here, because they would train
        # the model to emphasise the wrong words while every loss curve looks
        # perfectly normal.
        self._offset_tokenizer = None
        if self.sentence_labels is not None and not self.tokenizer.is_fast:
            self._offset_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, use_fast=True, trust_remote_code=True)
            self._offset_tokenizer.add_special_tokens(special_token)

        # 기존 구현과의 일관성을 유지
        '''
        NOTE:
        Qwen과 LLaMA의 pad_token_id, bos_token_id, eos_token_id가 동일한지 확인 필요.
        → 완전히 다릅니다!
        '''
        self.max_seq = max_seq
        self.image_folder = image_folder
        self.report_folder = report_folder
        self.z_chunk_size = z_chunk_size
        self.z_chunk_stride = z_chunk_stride
        self.xy_size = xy_size
        self.image_num = image_num

        # Load the JSONL data
        self.df = pd.read_json(json_path, lines=True)
        full_len = len(self.df)

        # Optional fixed subset of the split, used by the small-scale ablation
        # runs (configs/train/qwen3_ablation.sh) to get a signal in hours
        # instead of days. The draw is deterministic in ``subsample_seed``
        # *alone* -- not in anything that varies between ablation arms and not
        # in torch's global RNG, which train.py seeds but the arms consume
        # differently (an arm without a cls template draws one fewer random
        # number per sample). Every arm therefore sees a byte-identical subset
        # whatever its visual-token count or template flags, which is the whole
        # point of an ablation; sorting the drawn indices additionally keeps
        # the split's original ordering.
        if subsample_fraction is None:
            subsample_fraction = 1.0
        if not 0.0 < subsample_fraction <= 1.0:
            raise ValueError(
                f"subsample_fraction must satisfy 0 < f <= 1, got {subsample_fraction}"
            )
        if subsample_fraction < 1.0:
            keep = max(1, int(round(full_len * subsample_fraction)))
            index = np.sort(np.random.default_rng(subsample_seed).permutation(full_len)[:keep])
            self.df = self.df.iloc[index].reset_index(drop=True)
            print(
                f'Number of CT samples from {json_path}: {len(self.df)} of {full_len} '
                f'(subsample_fraction={subsample_fraction}, subsample_seed={subsample_seed})'
            )
        else:
            print('Number of CT training samples:', len(self.df))

        # Every sample this split will yield must have spans, checked once here
        # rather than discovered per sample during training. A labels file built
        # for a different split would otherwise leave part of the corpus at
        # uniform weights -- a run at a silently different objective, with a
        # perfectly normal-looking loss curve.
        if self.sentence_labels is not None:
            missing = [n for n in self.df['sample_name'].tolist()
                       if n not in self.sentence_labels]
            if missing:
                raise ValueError(
                    f"{len(missing)} of {len(self.df)} samples are absent from "
                    f"{sentence_labels_path} (e.g. {missing[:3]}); the spans file "
                    f"does not match this split")
            print(f'[token weight] spans cover all {len(self.df)} samples')

        # Cache
        rve_df = pd.read_csv(rve_manifest_path)
        self.rve_manifest = rve_df.set_index('sample_name')['image_cache_path'].to_dict()

        # 'chatml' for Qwen/ChatGLM/Gemma/Mistral etc., 'human_design' for Llama-2.
        self.message_type = _resolve_message_type(tokenizer_path, message_type)
        if self.sentence_labels is not None and 'chatml' not in self.message_type:
            # Only the chatml branch of __getitem__ emits token_weight. Refusing
            # here beats quietly running unweighted after the caller asked for
            # weighting.
            raise NotImplementedError(
                f"token weighting is implemented for the chatml prompt path only, "
                f"but message_type resolved to {self.message_type!r}")
        if self.message_type == 'human_design':
            # Llama-2 Chinese Chat ships without bos/eos/pad tokens set.
            self.tokenizer.pad_token_id = 0
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2

        if self.use_cls_template:
            self.cls_template_path = cls_template_path
            self.cls_report_template = _load_key_phrase_template(cls_template_path, "cls")

        if self.use_ret_template:
            self.ret_template_path = ret_template_path
            self.ret_report_template = _load_key_phrase_template(ret_template_path, "ret")

    def __len__(self) -> int:
        return len(self.df)

    def _text_add_ct_tokens(self, text: str) -> str:
        text = '<ct>' + self.image_padding_tokens[0] + '</ct>' + '。' + text
        return text

    def _token_weight(self, text, answer, sample_name, input_ids, label):
        """``[L]`` float weights aligned to `label`; see CTRG3D.forward.

        Offsets come from tokenizing the whole chat-templated `text`, not from a
        second tokenization of `answer` on its own: retokenizing the answer by
        itself shifts BPE boundaries at the seam with the prompt and would
        misplace every weight by a token. `answer` appears verbatim inside
        `text`, so its start is a plain string search and the sentence spans --
        which were measured against that same report_metadata string -- shift by
        that one constant.

        Raises rather than falling back to all-ones. The upstream implementation
        this idea comes from returns uniform weights on a miss and only bumps a
        counter, which turns a broken labels file into a run that trains at the
        wrong objective and reports nothing; this repo's convention is to fail
        early and loudly instead (cf. the vocabulary-size check in CTRG3D and
        the early-stopping guard in train.py). Coverage of every sample_name is
        already verified once at construction, so reaching either branch here
        means the answer text and the spans file disagree.
        """
        w = torch.ones_like(label, dtype=torch.float32)
        entry = self.sentence_labels.get(sample_name)
        if entry is None:
            raise KeyError(
                f"{sample_name} is not in the sentence-labels file; token "
                f"weights cannot be placed for it")
        # rfind, not find: the answer is the last thing in the chat-templated
        # text, and a retrieved sentence in the prompt could in principle repeat
        # a prefix of it.
        base = text.rfind(answer)
        if base < 0:
            raise RuntimeError(
                f"{sample_name}: report_metadata does not appear verbatim in the "
                f"chat-templated prompt, so the sentence spans cannot be anchored")

        tk = self._offset_tokenizer or self.tokenizer
        enc = tk(text, max_length=self.max_seq, truncation=True,
                 padding='max_length', return_offsets_mapping=True)
        # The fast twin must reproduce the configured tokenizer exactly, or the
        # offsets index different tokens than the ones being weighted.
        if len(enc['input_ids']) != input_ids.numel() or \
                enc['input_ids'] != input_ids.tolist():
            raise RuntimeError(
                f'{sample_name}: offset tokenizer disagrees with the configured '
                f'tokenizer; token weights would be misaligned')
        off = torch.tensor(enc['offset_mapping'])
        starts, ends = off[:, 0], off[:, 1]
        # Padding and special tokens carry offset (0, 0); `label != -100`
        # already excludes them, and every real span starts after the prompt so
        # `ends > a` cannot hold for them either.
        supervised = label != -100
        for span in entry['spans']:
            s, e, flag = span[0], span[1], span[2]
            if not flag:
                continue
            a, b = base + s, base + e
            # Half-open overlap: a token counts as inside the sentence if any
            # of its characters are, which is what makes the seam tokens at a
            # sentence boundary belong to the finding rather than to the
            # boilerplate before it.
            w[(starts < b) & (ends > a) & supervised] = self.label_token_weight
        return w

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[int(index)]
        sample_name = row.get('sample_name', str(index))

        # VISION
        rve_path = _resolve_rve_path(self.rve_manifest.get(sample_name), sample_name)
        ct_tensor = load_rve_volume(
            rve_path,
            target_slices=self.target_d,
            target_h=self.target_hw,
            target_w=self.target_hw,
            pad_value=self.pad_value,
            crop_mode='random' if self.is_train else 'center',
        )

        if ct_tensor.shape[-2] != ct_tensor.shape[-1]:
            ct_tensor = ct_tensor.permute(0, 3, 1, 2)  # DHW로 만들기 위함.

        # Ensure proper shape (add channel dimension if needed)
        if len(ct_tensor.shape) == 3:
            ct_tensor = ct_tensor.unsqueeze(0)

        # flip the ct_tensor along the first dimension
        ct_tensor = torch.flip(ct_tensor, dims=[1])

        # Skip windowing and normalization - will be done on GPU after collation, This is more efficient for batched operations

        # make it float32
        ct_tensor = ct_tensor.to(torch.float32)

        # TEXT
        answer = row["report_metadata"]

        cls_template = (
            _lookup_template(self.cls_report_template, sample_name, "cls")
            if self.use_cls_template
            else None
        )

        ret_template = (
            _lookup_template(self.ret_report_template, sample_name, "ret")
            if self.use_ret_template
            else None
        )

        # 학습 시에만 dropout 적용.
        # 순서: 먼저 템플릿 전체를 통째로 제거할지 결정한 뒤, 살아남은
        # 템플릿에 대해서만 문장 단위 dropout을 적용한다. (이미 dropout된
        # 결과를 버리는 낭비를 없애고, 두 확률의 의미를 분리해서 유지한다.)
        #
        # These three probabilities are the regularizer that stops the model
        # from *depending* on a template being present: template_full_drop_prob
        # removes a whole block (so the aux-free prompt variant from
        # build_instruction() is seen during training, exactly as at inference
        # when a prediction is missing), and the per-sentence dropouts make it
        # treat individual auxiliary sentences as droppable evidence rather
        # than as text to copy. They are applied per block and with identical
        # values in every ablation arm -- an arm that differs in this would be
        # comparing regularization strength, not the templates themselves.
        # Note the deliberate consequence: with both templates enabled the
        # chance of seeing *no* aux block at all is 0.05^2, versus 0.05 in a
        # single-template arm. That is the same rule applied per block, which
        # is what keeps each block's "may be absent" signal equally strong.
        if self.is_train:
            if cls_template is not None and torch.rand(1).item() < self.template_full_drop_prob:
                cls_template = None
            if ret_template is not None and torch.rand(1).item() < self.template_full_drop_prob:
                ret_template = None

            if cls_template is not None:
                cls_template = sentence_dropout(cls_template, drop_prob=self.cls_dropout_prob)
            if ret_template is not None:
                ret_template = sentence_dropout(ret_template, drop_prob=self.ret_dropout_prob)

        instruction = build_instruction(cls_template=cls_template, ret_template=ret_template)

        if 'chatml' in self.message_type:
            # The content of user combines images and instruction. The image placeholders are replaced with actual image tokens in the model.
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    # NOTE: a real newline. This used to be "\\n" inside a
                    # non-raw f-string, which put the two characters
                    # backslash + n into the prompt instead of a line break.
                    "role": "user",
                    "content": f"<ct>{self.image_padding_tokens[0]}</ct>\n{instruction}"
                }
            ]

            # The answer is added to the messages for the assistant to learn from.
            messages.append({"role": "assistant", "content": answer})

            # Tokenization is handled here using the chat template.
            self.tokenizer.padding_side = "right"

            try:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": False,
                    "enable_thinking": self.enable_thinking,
                }
                text = self.tokenizer.apply_chat_template(**template_kwargs)
            except:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": False,
                }
                text = self.tokenizer.apply_chat_template(**template_kwargs)

            text_tensor = self.tokenizer(text, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            # Truncation here eats the END of the sequence, which is the answer --
            # the exact tokens the loss is computed on. Every loss curve still looks
            # normal while the target is being clipped, so this cannot be left silent.
            # max_seq is set from a full-split token-length measurement per arm
            # (scripts/verify_maxseq_newtemplate.py), so hitting the cap means the
            # measurement and the running config disagree: a config bug, not a rare
            # long sample. `== max_seq` also covers the exact-fit case, which is
            # indistinguishable from truncation and equally not worth training on.
            if int(attention_mask.sum()) >= self.max_seq:
                raise RuntimeError(
                    f"sample {sample_name!r} tokenizes to >= max_seq ({self.max_seq}) "
                    f"and was truncated, which clips the answer the loss is taken on. "
                    f"Re-measure prompt lengths for this arm's templates and raise "
                    f"max_seq; do not silence this.")
            # Find the start of the assistant's answer for loss calculation
            # We need to tokenize the user part of the prompt separately to find its length
            prompt_messages = messages[:-1]  # Exclude assistant's answer

            try:
                prompt_template_kwargs = {
                    "conversation": prompt_messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": self.enable_thinking,
                }
                prompt_text = self.tokenizer.apply_chat_template(**prompt_template_kwargs)
            except:
                prompt_template_kwargs = {
                    "conversation": prompt_messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                prompt_text = self.tokenizer.apply_chat_template(**prompt_template_kwargs)

            prompt_tensor = self.tokenizer(prompt_text, max_length=self.max_seq, truncation=True, return_tensors="pt")
            prompt_len = torch.sum(prompt_tensor["attention_mask"][0])

            label = input_ids.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            # Mask out the prompt part, only calculate loss on the answer
            label[:prompt_len] = -100
            # Also mask special image tokens if they are in the labels
            label[label >= self.core_vocab_size] = -100

            out = {
                'lang_x': input_ids,
                'vision_x': {'ct_image': ct_tensor},   # (1, 3, H, W, D) - 원본 이미지, 분할 없음 // [1, 256, 256, 256]
                'mask_x': {},
                'region2area': {},
                'attention_mask': attention_mask,
                'label': label,
            }
            if self.sentence_labels is not None:
                out['token_weight'] = self._token_weight(
                    text, answer, sample_name, input_ids, label)
            return out

        elif 'human_design' in self.message_type:
            prompt = self._text_add_ct_tokens(instruction)

            self.tokenizer.padding_side = "right"
            text_tensor = self.tokenizer(prompt + ' ' + answer, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            effective_length = torch.sum(attention_mask)
            if effective_length < self.max_seq:
                input_ids[effective_length] = self.tokenizer.eos_token_id
            else:
                input_ids[self.max_seq - 1] = self.tokenizer.eos_token_id

            prompt_tensor = self.tokenizer(prompt, max_length=self.max_seq, truncation=True, padding="max_length", return_tensors="pt")
            prompt_len = torch.sum(prompt_tensor["attention_mask"][0])

            label = input_ids.clone()
            label[label == self.tokenizer.pad_token_id] = -100
            label[label >= self.core_vocab_size] = -100
            label[:prompt_len] = -100

            return {
                'lang_x': input_ids,
                'vision_x': {'ct_image': ct_tensor},  # (1, 3, H, W, D) - 원본 이미지, 분할 없음
                'mask_x': {},
                'region2area': {},
                'attention_mask': attention_mask,
                'label': label,
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test PETCTDataset_Train with local paths.")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--report_folder", required=True)
    parser.add_argument("--template_path", default=None)
    parser.add_argument("--use_template", action="store_true")
    parser.add_argument("--use_fast", action="store_true")
    args = parser.parse_args()

    dataset = CTRATEDataset_Train(
        tokenizer_path=args.tokenizer_path,
        image_folder=args.image_folder,
        report_folder=args.report_folder,
        template_path=args.template_path,
        use_template=args.use_template,
        use_fast=args.use_fast,
    )
    prompt_len_list = []
    for data in dataset:
        prompt_len_list.append(torch.sum(data["attention_mask"]).item())

    print(max(prompt_len_list))