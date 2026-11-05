import warnings

import numpy as np
import torch
import torch.nn.functional as F

from pillar.metrics.abstract import AbstractMetric
# 중복 제거: matcher는 모델의 staticmethod를 재사용 (코사인 기반, 식 2와 정합)
from pillar.models.heads.transq import Simple_TranSQ


class RetrievalMetric(AbstractMetric):
    _warned_about_default_censoring = False

    """
    TranSQ semantic-query 모델의 gallery retrieval 성능을 세 가지 관점에서 계산.

    1) match_recall@K / mrr / mean_rank / median_rank (기존):
       Hungarian 매칭이 실제로 짝지어준 (query, 정답 문장) 쌍마다, 그 정답 문장이
       gallery 전체에서 몇 등으로 채점되는지. matcher의 선택(모델 자신의 topic_preds
       확신도가 비용에 포함됨)에 종속적이라, "매칭이 된 것들 중" 품질만 본다는 한계가 있음.

    2) sample_recall@K (신규, matching에 안 걸려도 평가됨):
       한 환자의 실제 key phrase 집합 중, 모델이 **confidence(topic_preds) 상위 K개
       query**로 gallery에서 찾아낸 phrase 집합과 몇 개나 겹치는지. Hungarian 매칭 결과와
       무관하게 "모델이 자신 있어 하는 top-K를 믿으면 실제 소견의 몇 %를 잡아내는가"를
       바로 답해주는, 가장 직관적인 rank1/rank5류 지표.

    3) report_precision / report_recall / report_f1 (신규):
       실제 추론(Simple_TranSQ.retrieve())과 동일하게 prob > threshold인 query만 골라
       report를 만들었을 때, 실제 report(정답 phrase 집합) 대비 precision/recall/F1.

    Args:
        gallery_embeddings: (N_gallery, d_target) L2-normalized sentence gallery.
        k_values: recall@k cutoff 목록 (match_recall@K, sample_recall@K 공통).
        select_threshold: report_precision/recall/f1 계산 시 사용할 선택 임계값
            (Simple_TranSQ.retrieve()의 기본값 0.4와 동일하게 맞춤).
    """

    def __init__(self, args, dataset_info, split=None, **kwargs):
        super().__init__(args)
        self.dataset_info = dataset_info
        self.split = split

        # TranSQLoss와 동일한 키 컨벤션
        self.logit_key = kwargs.get("logit_key", "retrieval")
        self.sent_feats_key = kwargs.get("sent_feats_key", "sent_feats")
        self.topic_preds_key = kwargs.get("topic_preds_key", "topic_preds")
        self.gt_embeds_key = kwargs.get("target_embeds_key", "sent_embeds")
        self.gt_len_key = kwargs.get("target_len_key", "sent_num")
        self.k_values = kwargs.get("k_values", [1, 5, 10])
        self.match_mu = kwargs.get("match_mu", 0.5)  # loss/model과 동일한 μ
        self.select_threshold = kwargs.get("select_threshold", 0.4)  # Simple_TranSQ.retrieve()와 동일

        # gallery 매칭을 어디서 계산할지. None(기본) = CUDA가 보이면 CUDA, 없으면 입력 텐서의
        # device. 엔진이 gather한 예측을 CPU로 내려주기 때문에(to_cpu_recursive) 예전에는
        # 이 무거운 matmul이 전부 CPU에서 돌았다 — 실측(job 27887049 epoch 0): val 1,552
        # 샘플에 **17분 이상**, 그동안 GPU utilization 0%. 샘플당
        # (N_q=64, D=4096) @ (D, N_gallery=79,166) = 2.1e10 MAC 이고 이게 샘플 수만큼 반복된다.
        # 수식은 그대로이고 계산 위치만 옮기는 것이다(TF32는 이 저장소에서 켜지 않으므로
        # fp32 matmul 그대로 — 아래 _gallery_on() 참고).
        self.compute_device = kwargs.get("compute_device", None)

        gallery_embeddings = kwargs.get("gallery_embeddings", None)
        gallery_embeddings = torch.load(gallery_embeddings)
        # AMP 대비 float32로 캐스팅
        self.gallery_embeddings = F.normalize(gallery_embeddings.float(), p=2, dim=-1)
        # device로 옮긴 gallery 캐시. 매 validation마다 1.3GB(79,166 x 4096 fp32)를
        # 다시 전송하지 않기 위한 것.
        self._gallery_cache = {}

    def _resolve_device(self, fallback):
        if self.compute_device is not None:
            return torch.device(self.compute_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return fallback

    def _gallery_on(self, device):
        key = str(device)
        cached = self._gallery_cache.get(key)
        if cached is None:
            cached = self.gallery_embeddings.to(device)
            self._gallery_cache[key] = cached
        return cached

    @property
    def metric_keys(self):
        return {"target_label": [self.gt_embeds_key, self.gt_len_key],
                "pred_label": self.logit_key}

    def __call__(self, **kwargs):
        """
        Args:
            batch: gt_embeds_key, gt_len_key를 포함하는 배치 딕셔너리
            model_output: logit_key 아래에 sent_feats_key, topic_preds_key를 포함

        Returns:
            stats_dict: match_recall@k/mrr/mean_rank/median_rank, sample_recall@k,
                        report_precision/recall/f1
        """
        stats_dict = {}

        sent_feats, topic_preds = kwargs[self.logit_key]  # (B, N_q, D), (B, N_q)

        # aux_loss=True로 학습한 모델이 (L, B, N_q, ·)를 반환하면 최종 층만 평가에 사용
        if sent_feats.dim() == 4:
            sent_feats = sent_feats[-1]
            topic_preds = topic_preds[-1]

        gt_embeds = kwargs[self.gt_embeds_key]   # (B, N_max, D)
        gt_lens = kwargs[self.gt_len_key]        # (B,)

        # eval 시에는 NEFTune 노이즈를 적용하지 않는다 (gt_embeds 원본 사용).
        indices = Simple_TranSQ.hungarian_match(sent_feats, topic_preds, gt_embeds, gt_lens, mu=self.match_mu)

        # Hungarian 매칭(위)은 scipy 기반이라 CPU에 남기고, 아래 gallery 매칭만 device로
        # 옮긴다 — 여기가 이 metric 비용의 사실상 전부다(__init__의 compute_device 주석 참고).
        device = self._resolve_device(sent_feats.device)
        gallery = self._gallery_on(device)  # (N_gallery, D)
        max_k = max(self.k_values)

        prob = topic_preds.float().sigmoid().to(device)                    # (B, N_q)
        pred_norm_all = F.normalize(sent_feats.detach().float().to(device), p=2, dim=-1)  # (B, N_q, D)
        gt_embeds_dev = gt_embeds.detach().float().to(device)              # (B, N_max, D)
        conf_order = prob.argsort(dim=-1, descending=True)                # (B, N_q), confidence 내림차순 query idx

        ranks = []
        sample_recalls = {k: [] for k in self.k_values}
        precisions, report_recalls, f1s = [], [], []

        for i, (pred_idx, gt_idx) in enumerate(indices):
            n_tgt = int(gt_lens[i])
            if n_tgt == 0:
                continue

            # 이 샘플의 모든 query -> gallery 유사도를 한 번만 계산해서 세 metric 모두에 재사용
            sim_all_gallery = pred_norm_all[i] @ gallery.T             # (N_q, N_gallery)
            pred_gallery_idx_all = sim_all_gallery.argmax(dim=-1)      # (N_q,) query별 최근접 gallery id

            gt = F.normalize(gt_embeds_dev[i, :n_tgt], p=2, dim=-1)               # (n_tgt, D)
            true_gallery_idx = set((gt @ gallery.T).argmax(dim=-1).tolist())      # 실제 소견들의 gallery id 집합

            # ---- 1) match_recall@K / mrr / mean_rank / median_rank (Hungarian 매칭 쌍 기준) ----
            if len(pred_idx) > 0:
                sim_pred_gallery = sim_all_gallery[pred_idx]                       # (M, N_gallery)
                gt_matched = F.normalize(gt_embeds_dev[i, gt_idx], p=2, dim=-1)
                gt_gallery_idx = (gt_matched @ gallery.T).argmax(dim=-1)           # (M,)
                # 예전엔 매칭 쌍마다 .item()을 불러 쌍 수만큼 device 동기화가 났다(샘플당 최대
                # 64회 x 샘플 수). 같은 비교를 배치로 한 번에 하고 결과만 한 번 옮긴다 —
                # 부등호/합 연산이 동일하므로 순위 값도 동일하다.
                target_scores = sim_pred_gallery.gather(1, gt_gallery_idx.unsqueeze(1))  # (M, 1)
                ranks.extend(((sim_pred_gallery > target_scores).sum(dim=1) + 1).tolist())

            # ---- 2) sample_recall@K: confidence 상위 K개 query로 실제 소견의 몇 %를 잡아내는가 ----
            ranked_gallery = pred_gallery_idx_all[conf_order[i]].tolist()  # confidence 내림차순 gallery id
            for k in self.k_values:
                topk_hit = len(true_gallery_idx & set(ranked_gallery[:k]))
                sample_recalls[k].append(topk_hit / len(true_gallery_idx))

            # ---- 3) report_precision / report_recall / report_f1 (threshold 기반 선택, retrieve()와 동일 기준) ----
            selected = (prob[i] > self.select_threshold).nonzero(as_tuple=True)[0]
            if len(selected) == 0:
                selected = prob[i].argmax().unsqueeze(0)
            pred_report_idx = set(pred_gallery_idx_all[selected].tolist())
            tp = len(pred_report_idx & true_gallery_idx)
            precision = tp / len(pred_report_idx) if len(pred_report_idx) > 0 else 0.0
            recall = tp / len(true_gallery_idx)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            precisions.append(precision)
            report_recalls.append(recall)
            f1s.append(f1)

        if len(precisions) == 0:
            warnings.warn("RetrievalMetric: no samples with ground-truth sentences in this batch.")
            for k in self.k_values:
                stats_dict[f"match_recall@{k}"] = float("nan")
                stats_dict[f"sample_recall@{k}"] = float("nan")
            stats_dict["mrr"] = float("nan")
            stats_dict["mean_rank"] = float("nan")
            stats_dict["median_rank"] = float("nan")
            stats_dict["report_precision"] = float("nan")
            stats_dict["report_recall"] = float("nan")
            stats_dict["report_f1"] = float("nan")
            return stats_dict

        for k in self.k_values:
            stats_dict[f"sample_recall@{k}"] = float(np.mean(sample_recalls[k]))

        if len(ranks) > 0:
            ranks_arr = np.array(ranks)
            for k in self.k_values:
                stats_dict[f"match_recall@{k}"] = float(np.mean(ranks_arr <= k))
            stats_dict["mrr"] = float(np.mean(1.0 / ranks_arr))
            stats_dict["mean_rank"] = float(np.mean(ranks_arr))
            stats_dict["median_rank"] = float(np.median(ranks_arr))
        else:
            for k in self.k_values:
                stats_dict[f"match_recall@{k}"] = float("nan")
            stats_dict["mrr"] = float("nan")
            stats_dict["mean_rank"] = float("nan")
            stats_dict["median_rank"] = float("nan")

        stats_dict["report_precision"] = float(np.mean(precisions))
        stats_dict["report_recall"] = float(np.mean(report_recalls))
        stats_dict["report_f1"] = float(np.mean(f1s))

        return stats_dict