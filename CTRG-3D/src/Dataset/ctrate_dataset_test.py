import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from Dataset.ctrate_dataset_train import (
    SYSTEM_PROMPT,
    _load_key_phrase_template,
    _lookup_template,
    _resolve_message_type,
    _resolve_rve_path,
    build_instruction,
    load_rve_volume,
)


class CTRATEDataset_Test(Dataset):
    """CT-only inference dataset for CTRG3D, mirroring CTRATEDataset_Train.

    Kept deliberately in lock-step with CTRATEDataset_Train's tokenizer /
    special-token setup (same list, same add order) so that the input_ids
    produced here index the same embedding rows as at training time -- see
    the ordering note in Model.CTRG_3D.CTRG3D.__init__. Differences from the
    train dataset are inference-specific: no assistant turn / labels (uses
    add_generation_prompt=True instead), no cls/ret sentence dropout (always
    the full prediction, deterministic), and it carries sample_name /
    question / gt_report through for result logging.
    """

    def __init__(
        self,
        tokenizer_path: str,
        json_path: str,
        rve_manifest_path: str,
        inferenced_id: Optional[List[str]] = None,
        cls_template_path: str = None,
        ret_template_path: str = None,
        max_img_size: int = 1,
        image_num: int = 128,
        max_seq: int = 4096,
        use_fast: bool = False,
        use_cls_template: bool = False,
        use_ret_template: bool = False,
        enable_thinking: bool = False,
        message_type: str = None,
        target_d: int = 256,
        target_hw: int = 256,
        pad_value: int = -1024,
        subsample_fraction: float = 1.0,
        subsample_seed: int = 42,
        prompts_only: bool = False,
    ) -> None:
        super().__init__()
        # prompts_only: skip loading the CT volume in __getitem__ and return
        # vision_x['ct_image'] = None.  Only for callers that supply the visual
        # features from somewhere else (an offline vision cache) and read just
        # 'question' / 'gt_report' -- e.g. scripts/generate_batched.py and the
        # GRPO drivers.  Those read ~25 MB per sample and then discard it, which
        # is invisible on Biowulf storage but is 300 GB of pointless I/O over
        # storm's sshfs mount.  Default False keeps the old behaviour byte for
        # byte.  None (not an empty tensor) so that an accidental use fails
        # loudly instead of silently feeding zeros to the encoder.
        self.prompts_only = bool(prompts_only)
        self.target_d = target_d
        self.target_hw = target_hw
        self.pad_value = pad_value
        self.use_cls_template = use_cls_template
        self.use_ret_template = use_ret_template
        self.enable_thinking = enable_thinking
        self.max_seq = max_seq
        self.image_num = image_num

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast, trust_remote_code=True)

        # Must match CTRATEDataset_Train.__init__'s special_token list and add
        # order exactly (see Model.CTRG_3D.CTRG3D.__init__'s ordering note).
        special_token = {
            "additional_special_tokens": [
                "<ct>", "</ct>",
                "<cls_template>", "</cls_template>",
                "<ret_template>", "</ret_template>",
            ]
        }
        self.image_padding_tokens: List[str] = []
        for i in range(max_img_size):
            padding = ""
            for j in range(image_num):
                tok = "<image_ct" + str(i * image_num + j) + ">"
                padding += tok
                special_token["additional_special_tokens"].append(tok)
            self.image_padding_tokens.append(padding)
        self.tokenizer.add_special_tokens(special_token)

        self.message_type = _resolve_message_type(tokenizer_path, message_type)
        if self.message_type == 'human_design':
            self.tokenizer.pad_token_id = 0
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2

        self.df = pd.read_json(json_path, lines=True)
        full_len = len(self.df)

        # Same fixed subset the ablation arms were trained/evaluated against, so
        # inference can be run on the 388 validation samples instead of all
        # 1,552 (generation is autoregressive and ~4x the cost). The draw must
        # match CTRATEDataset_Train's exactly -- a permutation of the split
        # seeded by subsample_seed alone, sorted -- or the arms would be scored
        # on different data than they were evaluated on. Applied before the
        # resume filter so the target set does not shift between resumes.
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
                f'Number of CT test samples: {len(self.df)} of {full_len} '
                f'(subsample_fraction={subsample_fraction}, subsample_seed={subsample_seed})'
            )
        else:
            print('Number of CT test samples (before resume filtering):', len(self.df))

        already_done = set(inferenced_id or [])
        if already_done:
            self.df = self.df[~self.df['sample_name'].isin(already_done)].reset_index(drop=True)
            print('Remaining after skipping already-inferenced samples:', len(self.df))

        rve_df = pd.read_csv(rve_manifest_path)
        self.rve_manifest = rve_df.set_index('sample_name')['image_cache_path'].to_dict()

        if self.use_cls_template:
            self.cls_report_template = _load_key_phrase_template(cls_template_path, "cls")
        if self.use_ret_template:
            self.ret_report_template = _load_key_phrase_template(ret_template_path, "ret")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        row = self.df.iloc[int(index)]
        sample_name = row.get('sample_name', str(index))

        # VISION
        if self.prompts_only:
            ct_tensor = None
        else:
            rve_path = _resolve_rve_path(self.rve_manifest.get(sample_name), sample_name)
            ct_tensor = load_rve_volume(
                rve_path,
                target_slices=self.target_d,
                target_h=self.target_hw,
                target_w=self.target_hw,
                pad_value=self.pad_value,
                crop_mode='center',
            )

        if ct_tensor is not None and ct_tensor.shape[-2] != ct_tensor.shape[-1]:
            ct_tensor = ct_tensor.permute(0, 3, 1, 2)
        if ct_tensor is not None:
            if len(ct_tensor.shape) == 3:
                ct_tensor = ct_tensor.unsqueeze(0)
            ct_tensor = torch.flip(ct_tensor, dims=[1])
            ct_tensor = ct_tensor.to(torch.float32)

        # TEXT
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

        instruction = build_instruction(cls_template=cls_template, ret_template=ret_template)
        gt_report = row.get('report_metadata', '')

        if 'chatml' in self.message_type:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    # Real newline -- must stay byte-identical to
                    # CTRATEDataset_Train's user turn (see the note there).
                    "role": "user",
                    "content": f"<ct>{self.image_padding_tokens[0]}</ct>\n{instruction}"
                }
            ]

            try:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": self.enable_thinking,
                }
                text = self.tokenizer.apply_chat_template(**template_kwargs)
            except:
                template_kwargs = {
                    "conversation": messages,
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                text = self.tokenizer.apply_chat_template(**template_kwargs)

            text_tensor = self.tokenizer(text, max_length=self.max_seq, truncation=True, return_tensors="pt")
            lang_x = text_tensor["input_ids"][0]
            question_to_return = text

        elif 'human_design' in self.message_type:
            prompt = '<ct>' + self.image_padding_tokens[0] + '</ct>' + '。' + instruction
            text_tensor = self.tokenizer(prompt, max_length=self.max_seq, truncation=True, return_tensors="pt")
            lang_x = text_tensor["input_ids"][0]
            question_to_return = prompt

        return {
            'sample_name': sample_name,
            'lang_x': lang_x,
            'vision_x': {'ct_image': ct_tensor},
            'question': question_to_return,
            'gt_report': '' if gt_report is None else str(gt_report),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test CTRATEDataset_Test with local paths.")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--rve_manifest_path", required=True)
    parser.add_argument("--use_fast", action="store_true")
    args = parser.parse_args()

    dataset = CTRATEDataset_Test(
        tokenizer_path=args.tokenizer_path,
        json_path=args.json_path,
        rve_manifest_path=args.rve_manifest_path,
        use_fast=args.use_fast,
    )
    sample = dataset[0]
    print(sample['sample_name'])
    print(sample['question'])
