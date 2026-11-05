import warnings

import numpy as np
import torch
from lifelines.utils.btree import _BTree
from pillar.utils.logging import logger
import torch.nn.functional as F
from torchmetrics.functional import (
    auroc,
    precision_recall_curve,
    auc,
    average_precision,
)

from pillar.metrics.abstract import AbstractMetric
from pillar.datasets.nlst import CENSORING_DIST

from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)

LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]

class AbnMetric(AbstractMetric):
    _warned_about_default_censoring = False

    def __init__(self, args, dataset_info, split=None, **kwargs):
        super().__init__(args)
        self.dataset_info = dataset_info
        self.split = split
        self.logit_key = kwargs.get("logit_key", "logit")
        self.target_key = kwargs.get("target_key", "label")

    @property
    def metric_keys(self):
        return {"target_label": self.target_key, "pred_label": self.logit_key}

    def __call__(self, **kwargs):
        """
        Computes standard classification metrics

        Args:
            probs, preds, golds, censors: items from computing loss and model outputs

        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc

        Note:
            In multiclass setting (>2), accuracy, and micro-f1, micro-recall, micro-precision are equivalent
            Macro: calculates metric per class then averages
        """

        stats_dict = {}

        # Get required keys for current split
        y_true = kwargs[self.target_key].detach().float().cpu().numpy()
        logit  = kwargs[self.logit_key]
        y_pred = torch.sigmoid(logit).detach().float().cpu().numpy()

        p_all, r_all, f_all, a_all, auc_all = [], [], [], [], []
        for i, col in enumerate(LABEL_COLS):
            p = precision_score(y_true[:, i], y_pred[:, i].round(), zero_division=0)
            r = recall_score   (y_true[:, i], y_pred[:, i].round(), zero_division=0)
            f = f1_score       (y_true[:, i], y_pred[:, i].round(), zero_division=0)
            a = accuracy_score (y_true[:, i], y_pred[:, i].round())

            # 배치/에폭 내 해당 라벨이 단일 클래스만 존재하면 roc_auc_score가 에러를 던지므로 방어
            if np.unique(y_true[:, i]).size < 2:
                auc_score = float("nan")
            else:
                auc_score = roc_auc_score(y_true[:, i], y_pred[:, i])

            p_all.append(p); r_all.append(r); f_all.append(f); a_all.append(a)
            auc_all.append(auc_score)
            stats_dict[col+"_f1"] = float(np.mean(f))
            stats_dict[col+"_auroc"] = float(auc_score)

        stats_dict["macro_precision"] = float(np.mean(p_all))
        stats_dict["macro_recall"] = float(np.mean(r_all))
        stats_dict["macro_f1"] = float(np.mean(f_all))
        stats_dict["macro_accuracy"] = float(np.mean(a_all))
        stats_dict["macro_auroc"] = float(np.nanmean(auc_all))

        return stats_dict