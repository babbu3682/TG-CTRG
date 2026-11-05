from __future__ import annotations

import json
import re

import numpy as np
import torch

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from pillar.metrics.abstract import AbstractMetric


def safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower()


class MultiLabelClassificationMetric(AbstractMetric):
    def __init__(
        self,
        args,
        dataset_info=None,
        split=None,
        target_key="targets",
        logit_key="findings",
        label_names_path=None,
        **kwargs,
    ):
        super().__init__(args)

        self.target_key = target_key
        self.logit_key = logit_key
        self.label_names_path = label_names_path
        self.split = split

    @property
    def metric_keys(self):
        return {
            "target_label": self.target_key,
            "pred_label": self.logit_key,
        }

    def __call__(self, **kwargs):
        targets = kwargs[self.target_key]
        logits = kwargs[self.logit_key]

        targets = (
            targets.detach()
            .float()
            .cpu()
            .numpy()
        )

        probabilities = (
            torch.sigmoid(logits)
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if targets.ndim != 2:
            raise ValueError(
                f"Expected targets [N, C], got {targets.shape}"
            )

        num_classes = targets.shape[1]

        label_names = [
            f"class_{i:02d}"
            for i in range(num_classes)
        ]

        if self.label_names_path is not None:
            with open(
                self.label_names_path,
                "r",
                encoding="utf-8",
            ) as f:
                loaded = json.load(f)

            if len(loaded) != num_classes:
                raise ValueError(
                    f"label_names={len(loaded)}, "
                    f"num_classes={num_classes}"
                )

            label_names = loaded

        aurocs = []
        auprcs = []

        result = {}

        for i, label_name in enumerate(label_names):
            y_true = targets[:, i]
            y_prob = probabilities[:, i]

            # AUROC requires both positive and negative samples
            if len(np.unique(y_true)) < 2:
                auroc = np.nan
            else:
                auroc = roc_auc_score(
                    y_true,
                    y_prob,
                )

            # AP is meaningful when positives exist
            if y_true.sum() == 0:
                auprc = np.nan
            else:
                auprc = average_precision_score(
                    y_true,
                    y_prob,
                )

            aurocs.append(auroc)
            auprcs.append(auprc)

            key = safe_name(label_name)

            result[f"auroc_{key}"] = float(auroc)
            result[f"auprc_{key}"] = float(auprc)

        result["macro_auroc"] = float(
            np.nanmean(aurocs)
        )

        result["macro_auprc"] = float(
            np.nanmean(auprcs)
        )

        result["micro_auroc"] = float(
            roc_auc_score(
                targets.reshape(-1),
                probabilities.reshape(-1),
            )
        )

        result["micro_auprc"] = float(
            average_precision_score(
                targets.reshape(-1),
                probabilities.reshape(-1),
            )
        )

        return result