import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pillar.losses.abstract import AbstractLoss
from pillar.utils.logging import logger
# 중복 제거: matcher와 contrastive loss는 모델의 staticmethod를 재사용
from pillar.models.heads.transq import Simple_TranSQ


class TranSQLoss(AbstractLoss):
    def __init__(self, args, **kwargs):
        super().__init__(args)

        # ---- 키 매핑 ----
        self.logit_key = kwargs.get("logit_key", "retrieval")            # -> (sent_feats, topic_preds)
        self.sent_feats_key = kwargs.get("sent_feats_key", "sent_feats")
        self.topic_preds_key = kwargs.get("topic_preds_key", "topic_preds")
        self.gt_embeds_key = kwargs.get("target_embeds_key", "sent_embeds")  # (B, N_max, D)
        self.gt_len_key = kwargs.get("target_len_key", "sent_num")          # (B,)

        # ---- Loss 가중치 (논문 식 7: total = LPM + λ·LSC, LPM = sim + cls) ----
        self.sim_weight = kwargs.get("sim_weight", 1.0)
        self.cls_weight = kwargs.get("cls_weight", 1.0)
        self.contrastive_weight = kwargs.get("contrastive_weight", 0.1)   # λ (논문 4.3)

        # ---- Matcher ----
        self.match_mu = kwargs.get("match_mu", 0.5)                        # μ (논문 4.3)

        # ---- Classification loss 설정 ----
        # "db"(distribution-balanced, 논문 3.2.2 기본), "bce", "focal"
        self.cls_loss_type = kwargs.get("cls_loss_type", "db")
        self.num_queries = kwargs.get("num_queries", 64)
        # 논문 부록 D: positive class size = 데이터의 평균 key phrase 수,
        # negative class size = N - positive. 실측(2026-08-16, merged_*_prob_half.json의
        # "sentences" 길이): train 15.286 / valid 15.206, min 1 / max 49 / p95 22.
        self.avg_pos = kwargs.get("avg_pos_queries", 15.3)
        # DB 하이퍼파라미터 — COCO-MLT 설정 기반(논문 부록 D가 지목한 설정).
        self.neg_scale = kwargs.get("neg_scale", 2.0)                      # κ
        self.init_bias_scale = kwargs.get("init_bias", 0.05)              # ν 스케일
        self.db_focal = kwargs.get("db_focal", True)
        # re-balance weighting(R). 이전 구현은 이 항을 통째로 빼고 NTR만 뒀는데,
        # 부록 D의 "treated as single-label binary classification"은 **빈도값을 상수로
        # 고정하라**는 뜻이지 R을 빼라는 뜻이 아니다(아래 _db_loss 주석 참고).
        self.cls_rebalance = kwargs.get("cls_rebalance", True)
        self.balance_param = kwargs.get("balance_param", 2.0)
        self.map_alpha = kwargs.get("map_alpha", 0.1)
        self.map_beta = kwargs.get("map_beta", 10.0)
        # map_gamma(μ): "auto" 를 주면 1/avg_pos 로 두어 sigmoid의 중점을 우리 데이터의
        # 중앙값 샘플에 맞춘다. 기본은 COCO-MLT/논문 값 0.2 (아래 측정 주석 참고).
        _mg = kwargs.get("map_gamma", 0.2)
        self.map_gamma = (1.0 / self.avg_pos) if _mg == "auto" else float(_mg)
        # 일반 BCE/focal용
        self.pos_weight = kwargs.get("pos_weight", 5.0)
        self.gamma = kwargs.get("gamma", 2.0)
        self.alpha = kwargs.get("alpha", 0.75)

        # ---- Contrastive loss 설정 ----
        # ⚠️ contrastive_temp=1.0 은 실수가 아니라 **논문 식 4~6을 그대로 옮긴 것**이다.
        # RA-RRG 식 (4) Z_kl = v_k^T v̂_l, (5) q = softmax((S^v + S^v̂)/2), (6) 대칭 CE 어디에도
        # 온도 항이 없다. 결과적으로 이 항은 사실상 상수다 — job 27890344 실측: 30 epoch 동안
        # contrastive_loss 5.0139 -> 5.0126 (변화 0.003%), 즉 log|E|(|E|≈157.6, log=5.0603)의
        # 99.9%에 고정. cosine logit이 [-1,1]에 갇혀(diag 0.919 / off-diag 0.613) softmax가
        # 거의 균등해지고, soft target q도 같이 평평해지기 때문(q_max가 균등의 1.46배).
        # gradient 실측: sim_loss의 **0.1%** (temp 0.2 -> 12.6%, temp 0.07 -> 129%).
        #
        # **온도를 넣는 시도는 이미 검증됐고 실패했다.** 형제 구현
        # ct-rate-rrg/stage2_transq_rrg/exp/ct_rrg/model/modeling.py 의 run 4가 같은 λ=0.1에
        # tau=0.07을 줘서 sc_loss가 실제로 6.17 -> 3.89로 내려갔지만, **epoch 2~7 전 구간에서
        # retrieval이 더 나빴다(R@10 0.072 vs 대조군 0.099)** — 그래서 그쪽은 이 옵션을 코드에서
        # 아예 제거했다. 따라서 여기서도 기본값은 논문 그대로 1.0으로 두고, 바꾸고 싶으면
        # λ(contrastive_weight)를 같이 내리는 것을 전제로 할 것. 중간값(0.2)은 아직 아무도
        # 돌려본 적이 없다.
        self.contrastive_temp = kwargs.get("contrastive_temp", 1.0)
        self.contrastive_detach = kwargs.get("contrastive_detach", True)

        # ---- NEFTune 텍스트 임베딩 노이즈 (논문 3.2.1, 학습 시에만) ----
        self.use_neftune = kwargs.get("use_neftune", True)

        # ---- sim_loss 축약 방식 ----
        # "pair"(기본, 논문): 식 3이 샘플 안에서 Σ_{i=1..N}, 식 7이 Σ_{b=1..B} 이므로
        #   전체 sim 항은 "배치의 모든 매칭 쌍에 대한 합"이다 → 쌍마다 균등 가중.
        # "sample"(레거시, 2026-08-16 이전 런): 샘플별 평균을 낸 뒤 다시 평균 →
        #   쌍당 가중치가 1/n_b 가 되어 문장 수가 적은 볼륨이 과대 대표된다.
        #   실측(train 23,869): 논문 대비 상대 가중치가 n_pos=1 에서 13.7배,
        #   n_pos=49 에서 0.28배, p5~p95(9~22문장)만 봐도 1.53배 vs 0.63배로 벌어진다.
        # "query"(RA-RRG 식 3에 충실): 식 3은 **하나의 합** 안에서 L_cls를 i=1..N 전부에
        #   대해 더하고 (1 - L_sim)은 매칭된 쌍(y_i != empty)에만 더한다. 우리 구현의
        #   _classification_loss는 (B, N_q) 전체에 대한 mean(=Σ/(B·N))이므로, sim 항도
        #   같은 분모 B·N을 써야 두 항의 **상대 가중치**가 논문과 같아진다.
        #   "pair"(Σ/(B·n̄))는 sim을 N/n̄ 배 과대 가중한다 — 실측 N=64, n̄=15.286(train)
        #   이므로 **4.19배**. 즉 sim_weight=cls_weight=1.0으로 두면 논문 대비 selection
        #   loss가 4.19배 약하게 들어간다. 관측된 증상과 일치한다(cls_loss가 학습 내내
        #   +46~57% 악화, threshold 0.4에서 GT의 1.85배 과다 발화 → report FP 3,521).
        self.sim_reduction = kwargs.get("sim_reduction", "pair")
        if self.sim_reduction not in ("pair", "sample", "query"):
            raise ValueError(
                f"sim_reduction must be 'pair', 'sample' or 'query', got {self.sim_reduction!r}")

        # positive prior b, DB logit bias ν 사전 계산
        b = min(max(self.avg_pos / self.num_queries, 1e-6), 1 - 1e-6)
        # ν = logit(b) 스케일 (COCO-MLT: init_bias / neg_scale)
        self._db_bias = math.log(b / (1.0 - b)) * self.init_bias_scale / self.neg_scale

    # ---------------------------------------------------------------- #
    # 1. Similarity Loss (1 - cosine)  — 논문 식 3
    # ---------------------------------------------------------------- #
    def _similarity_loss(self, sent_feats, gt_embeds, indices):
        """식 3의 두 번째 항. 축약 방식은 self.sim_reduction 참고."""
        device = sent_feats.device
        if self.sim_reduction == "sample":
            sim_loss = torch.tensor(0.0, device=device)
            n_valid = 0
            for i, (pred_idx, gt_idx) in enumerate(indices):
                if len(pred_idx) > 0:
                    p_feat = F.normalize(sent_feats[i, pred_idx].float(), p=2, dim=-1)
                    g_feat = F.normalize(gt_embeds[i, gt_idx].float(), p=2, dim=-1)
                    sim_loss = sim_loss + (1.0 - (p_feat * g_feat).sum(dim=-1)).mean()
                    n_valid += 1
            return sim_loss / max(n_valid, 1)

        # "pair"/"query": 배치의 매칭 쌍을 전부 모아 한 번에 축약 → 쌍마다 균등 가중.
        # 분모만 다르다: "pair"는 쌍 개수, "query"는 B·N_q(= cls 항과 동일한 분모,
        # 논문 식 3의 상대 가중치). 위 self.sim_reduction 주석 참고.
        p_list, g_list = [], []
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                p_list.append(sent_feats[i, pred_idx])
                g_list.append(gt_embeds[i, gt_idx])
        if not p_list:
            return torch.tensor(0.0, device=device)
        p_feat = F.normalize(torch.cat(p_list, dim=0).float(), p=2, dim=-1)
        g_feat = F.normalize(torch.cat(g_list, dim=0).float(), p=2, dim=-1)
        per_pair = 1.0 - (p_feat * g_feat).sum(dim=-1)
        if self.sim_reduction == "query":
            return per_pair.sum() / (sent_feats.size(0) * self.num_queries)
        return per_pair.mean()

    # ---------------------------------------------------------------- #
    # 2. Classification Loss (DB / BCE / Focal)  — 논문 식 3의 L_cls
    # ---------------------------------------------------------------- #
    def _build_targets(self, topic_preds, indices):
        bs, num_query = topic_preds.shape
        tgt = torch.zeros_like(topic_preds)
        for i, (pred_idx, _) in enumerate(indices):
            if len(pred_idx) > 0:
                tgt[i, pred_idx] = 1.0
        return tgt

    def _classification_loss(self, topic_preds, indices):
        device = topic_preds.device
        tgt = self._build_targets(topic_preds, indices)  # (B, N_q)

        if self.cls_loss_type == "db":
            return self._db_loss(topic_preds, tgt)

        if self.cls_loss_type == "focal":
            pos_w = torch.tensor([self.pos_weight], device=device)
            bce = F.binary_cross_entropy_with_logits(
                topic_preds, tgt, pos_weight=pos_w, reduction="none")
            p = torch.sigmoid(topic_preds)
            p_t = p * tgt + (1 - p) * (1 - tgt)
            alpha_t = self.alpha * tgt + (1 - self.alpha) * (1 - tgt)
            loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
            return loss.mean()

        # 기본 BCE
        pos_w = torch.tensor([self.pos_weight], device=device)
        return F.binary_cross_entropy_with_logits(
            topic_preds, tgt, pos_weight=pos_w, reduction="mean")

    def _db_loss(self, logits, labels):
        """Distribution-Balanced Loss (Wu et al., 2020) — 논문 3.2.2의 selection loss.

        re-balance weighting(R) + negative-tolerant regularization(NTR) 두 항으로 이뤄진다.
        **이전 구현은 R을 통째로 빼고 NTR만 뒀는데, 그 근거로 달아둔 주석이 틀렸다** —
        부록 D의 "The selection process is treated as single-label binary classification,
        with the positive class size set to 7.16 ... and the negative class size fixed at
        N - 7.16" 은 R에 들어갈 **빈도값을 상수로 고정하라**는 뜻이지 R을 빼라는 뜻이 아니다.

        빈도를 상수 n 으로 고정하면 R이 이렇게 축약된다:
            freq_inv    = 1/n                          (모든 slot 동일)
            repeat_rate = Σ_{i∈y} 1/n = n_pos / n
            r           = freq_inv / repeat_rate = 1/n_pos     ← 그 **샘플의** 문장 수
            r_hat       = α + sigmoid(β·(r − μ))               (α=0.1, β=10, μ=0.2)
        즉 상수 빈도에서도 R은 사라지지 않고 "문장이 적은 볼륨을 더 세게 본다"는
        per-sample 가중치로 남는다.

        ⚠️ 다만 **우리 타깃 분포에서는 이 항이 거의 상수다.** 실측(train 23,869 볼륨,
        n_pos 평균 15.3 / p5~p95 = 9~22):
            μ=0.2 (COCO-MLT/논문)   r_hat p5~p95 = 0.276~0.391 (1.42배), 평균 0.320
            μ=1/15.3 ("auto")        r_hat p5~p95 = 0.550~0.712 (1.29배)  ← 오히려 더 평평
        n_pos 가 커지면 1/n_pos 가 좁은 구간(0.045~0.111)에만 머물러 sigmoid의 평평한
        왼쪽 꼬리에 갇히기 때문이다. 논문 레짐(n_pos≈7.16)은 r≈0.14 로 sigmoid의 상승
        구간에 걸쳐 2.64배까지 벌어진다 — 문장 단위 타깃을 쓰는 우리와 key phrase 단위를
        쓰는 논문의 차이가 여기서 드러난다.
        따라서 실질 효과는 (a) cls_loss 전체를 balance_param·r_hat ≈ 0.64 배로 재스케일
        (cls_weight 가 흡수하는 성분) 과 (b) n_pos ≤ 5 인 소수(약 1%) 볼륨에 대한 실제
        가중 상향, 두 가지다. μ/β 를 우리 레짐으로 다시 튜닝하고 싶으면
        map_gamma="auto" / map_beta 를 올리면 되지만, 근거 없이 두 상수를 동시에 논문에서
        떼어놓는 것이므로 기본값은 논문 값 그대로 둔다.
        """
        # R: 상수 빈도에서 r = 1/n_pos (샘플별). n_pos=0 방어로 1 이상으로 clamp.
        n_pos = labels.sum(dim=1, keepdim=True).clamp_min(1.0)     # (B, 1)
        if self.cls_rebalance:
            r_hat = self.map_alpha + torch.sigmoid(self.map_beta * (1.0 / n_pos - self.map_gamma))
        else:
            r_hat = torch.ones_like(n_pos)

        # NTR: 편향 ν를 **먼저** 더한 뒤 음성 로짓을 κ로 스케일한다(공식 구현 순서).
        # 이전 구현은 스케일 후에 더해서 음성 쪽 편향이 κ배 작게 들어가 있었다.
        z = logits + self._db_bias
        z = z * labels + z * (1 - labels) * self.neg_scale

        # 음성 샘플 가중치를 1/κ로 하향 (R과 곱해짐)
        w = r_hat * labels + (r_hat / self.neg_scale) * (1 - labels)

        bce = F.binary_cross_entropy_with_logits(z, labels, reduction="none")

        if self.db_focal:
            p = torch.sigmoid(z)
            p_t = p * labels + (1 - p) * (1 - labels)
            bce = ((1 - p_t) ** self.gamma) * bce

        return self.balance_param * (w * bce).mean()

    # ---------------------------------------------------------------- #
    # 3. In-batch Semantic Contrastive Loss  — 논문 식 4~6
    # ---------------------------------------------------------------- #
    def _contrastive_loss(self, sent_feats, gt_embeds, indices):
        """
        Hungarian 매칭으로 얻은 positive pair (v_i: text embed, v̂_σ(i): semantic embed)를
        배치 전체에서 flatten 하여 CLIP-style 대칭 대조 손실을 계산.

        Z_kl = v_k^T v̂_l                                 (식 4)
        q    = softmax((v v^T + v̂ v̂^T)/2)               (식 5, soft target)
        L_SC = (1/2|E|) Σ_k [ H(q_k,:, softmax(Z_k,:))
                            + H(q_:,k, softmax(Z_:,k)) ]  (식 6)

        Note:
            식 (6)의 두 번째 항은 열 방향 CE. M이 대칭이므로 열 target은 softmax(M, dim=0).
        """
        device = sent_feats.device
        v_hat_list, v_list = [], []
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                v_hat_list.append(sent_feats[i, pred_idx])   # semantic (v̂)
                v_list.append(gt_embeds[i, gt_idx])          # text (v)

        # positive pair가 1개 이하이면 대조 손실 정의 불가 → 0
        if len(v_hat_list) == 0:
            return torch.tensor(0.0, device=device)

        v_hat = F.normalize(torch.cat(v_hat_list, dim=0).float(), p=2, dim=-1)
        v = F.normalize(torch.cat(v_list, dim=0).float(), p=2, dim=-1)
        if v.size(0) < 2:
            return torch.tensor(0.0, device=device)

        Z = v @ v_hat.T                                   # (식 4)

        M = (v @ v.T + v_hat @ v_hat.T) / 2.0             # (식 5)
        if self.contrastive_detach:
            M = M.detach()
        q_row = F.softmax(M, dim=1)
        q_col = F.softmax(M, dim=0)

        logits = Z / self.contrastive_temp
        loss_row = -(q_row * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss_col = -(q_col * F.log_softmax(logits, dim=0)).sum(dim=0).mean()
        return 0.5 * (loss_row + loss_col)                # (식 6)

    # ---------------------------------------------------------------- #
    # 단일 층에 대한 손실 계산
    # ---------------------------------------------------------------- #
    def _compute_single(self, sent_feats, topic_preds, gt_embeds, gt_lens):
        indices = Simple_TranSQ.hungarian_match(sent_feats, topic_preds, gt_embeds, gt_lens, mu=self.match_mu)
        sim_loss = self._similarity_loss(sent_feats, gt_embeds, indices)
        cls_loss = self._classification_loss(topic_preds, indices)
        con_loss = self._contrastive_loss(sent_feats, gt_embeds, indices)
        return sim_loss, cls_loss, con_loss

    # ---------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------- #
    def __call__(self, batch=None, model_output=None, **extras):
        sent_feats, topic_preds = model_output[self.logit_key]
        gt_embeds = batch[self.gt_embeds_key]   # (B, N_max, D)
        gt_lens = batch[self.gt_len_key]        # (B,)

        train_flag = extras.get("train", None)
        if train_flag is None:
            train_flag = getattr(self, "training", None)
        if train_flag is None:
            raise RuntimeError("TranSQLoss requires an explicit train/eval flag for NEFTune noise.")
        is_train = bool(train_flag)        
                              
        # NEFTune: 학습 시에만 텍스트 임베딩에 노이즈 주입 후 L2 정규화.
        #   matcher / sim / contrastive 가 모두 '동일하게' 노이즈된 임베딩을 쓰도록
        #   여기서 한 번만 적용한다. eval 시엔 정규화만.
        if self.use_neftune:
            gt_embeds = Simple_TranSQ.add_neftune_noise(gt_embeds, training=is_train, normalize=True)

        # aux_loss: 모델이 (L, B, N_q, ·)를 반환하면 층별로 합산
        if sent_feats.dim() == 4:
            L = sent_feats.size(0)
            sim_loss = cls_loss = con_loss = 0.0
            for l in range(L):
                s, c, k = self._compute_single(
                    sent_feats[l], topic_preds[l], gt_embeds, gt_lens)
                sim_loss = sim_loss + s
                cls_loss = cls_loss + c
                con_loss = con_loss + k
            sim_loss, cls_loss, con_loss = sim_loss / L, cls_loss / L, con_loss / L
        else:
            sim_loss, cls_loss, con_loss = self._compute_single(sent_feats, topic_preds, gt_embeds, gt_lens)

        total_loss = (self.sim_weight * sim_loss
                      + self.cls_weight * cls_loss
                      + self.contrastive_weight * con_loss)

        logging_dict = OrderedDict({
            "sim_loss": sim_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "contrastive_loss": con_loss.detach(),
            "total_loss": total_loss.detach(),
        })
        return total_loss, logging_dict

    @property
    def loss_keys(self):
        return {"target_label": [self.gt_embeds_key, self.gt_len_key], 
                "pred_label": self.logit_key}
    