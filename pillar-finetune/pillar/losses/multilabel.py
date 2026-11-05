from __future__ import annotations

import json
from collections import OrderedDict

import torch
import torch.nn.functional as F

from pillar.losses.abstract import AbstractLoss


class MultiLabelBCELoss(AbstractLoss):
    _pos_weight_cache = {}

    def __init__(
        self,
        args,
        target_key="targets",
        logit_key="findings",
        pos_weight_path=None,
        max_pos_weight=None,
        **kwargs,
    ):
        super().__init__(args)

        self.target_key = target_key
        self.logit_key = logit_key
        self.pos_weight_path = pos_weight_path
        self.max_pos_weight = max_pos_weight

    def _get_pos_weight(self, device):
        if self.pos_weight_path is None:
            return None

        cache_key = (
            self.pos_weight_path,
            self.max_pos_weight,
        )

        if cache_key not in self._pos_weight_cache:
            with open(
                self.pos_weight_path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            values = data["pos_weight"]

            pos_weight = torch.tensor(
                values,
                dtype=torch.float32,
            )

            if self.max_pos_weight is not None:
                pos_weight = torch.clamp(
                    pos_weight,
                    max=float(self.max_pos_weight),
                )

            self._pos_weight_cache[cache_key] = pos_weight

        return self._pos_weight_cache[cache_key].to(device)

    def __call__(
        self,
        batch=None,
        model_output=None,
        **extras,
    ):
        logits = model_output[self.logit_key]
        targets = batch[self.target_key].float()

        if logits.shape != targets.shape:
            raise ValueError(
                f"logits/targets shape mismatch: "
                f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
            )

        pos_weight = self._get_pos_weight(
            logits.device
        )

        loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            targets.float(),
            pos_weight=pos_weight,
            reduction="mean",
        )

        logging_dict = OrderedDict()
        logging_dict["multilabel_bce"] = loss.detach()

        return loss, logging_dict

    @property
    def loss_keys(self):
        return {
            "target_label": self.target_key,
            "pred_label": self.logit_key,
        }