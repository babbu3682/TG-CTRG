from collections import OrderedDict

import torch
import torch.nn.functional as F

from pillar.losses.abstract import AbstractLoss


class MultiLabelWeightedBCELoss(AbstractLoss):
    """
    Multi-label weighted BCE loss (순수 pos_weight 보정, focal/alpha 미적용).

    - pos_weight (병변별 neg/pos): 클래스별 양/음성 불균형 보정

    최종 loss = mean( BCE_with_posweight )
    """

    # ------------------------------------------------------------------ #
    #  CT-RATE (47,149 samples) 실측 pos_weight (neg/pos)
    #  유병률 범위 7.2% ~ 45.4%, pos_weight 범위 1.21 ~ 12.82 → 클리핑 불필요
    #  CTRATE_LABEL_COLUMNS 순서와 동일하게 정렬되어 있어야 함
    # ------------------------------------------------------------------ #
    CTRATE_POS_WEIGHT = [
        7.10,   # Medical material                (prev 12.34%)
        2.52,   # Arterial wall calcification     (prev 28.37%)
        7.88,   # Cardiomegaly                    (prev 11.26%)
        12.82,  # Pericardial effusion            (prev  7.24%)
        2.92,   # Coronary artery wall calcif.    (prev 25.50%)
        5.98,   # Hiatal hernia                   (prev 14.32%)
        2.86,   # Lymphadenopathy                 (prev 25.92%)
        4.17,   # Emphysema                       (prev 19.35%)
        2.84,   # Atelectasis                     (prev 26.01%)
        1.21,   # Lung nodule                     (prev 45.35%)
        1.71,   # Lung opacity                    (prev 36.95%)
        2.75,   # Pulmonary fibrotic sequela      (prev 26.70%)
        7.26,   # Pleural effusion                (prev 12.10%)
        11.96,  # Mosaic attenuation pattern      (prev  7.72%)
        8.48,   # Peribronchial thickening        (prev 10.55%)
        4.67,   # Consolidation                   (prev 17.64%)
        8.96,   # Bronchiectasis                  (prev 10.04%)
        11.59,  # Interlobular septal thickening  (prev  7.94%)
    ]

    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.logit_key  = kwargs.get("logit_key", "logit")
        self.target_key = kwargs.get("target_key", "label")

        pos_weight = kwargs.get("pos_weight", None)  # 병변별 neg/pos 보정

        # pos_weight 미지정 시 CT-RATE 실측값을 기본 사용
        if pos_weight is None:
            pos_weight = torch.as_tensor(self.CTRATE_POS_WEIGHT, dtype=torch.float32)
        else:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)

        assert pos_weight.numel() == len(self.CTRATE_POS_WEIGHT), (
            f"pos_weight must have {len(self.CTRATE_POS_WEIGHT)} elements, "
            f"got {pos_weight.numel()}"
        )

        # 이 데이터셋은 최대 12.82라 클리핑 불필요. 다른 분포에서만 선택적으로 사용.
        self.pos_weight_max = kwargs.get("pos_weight_max", None)  # 예: 다른 데이터에서 10.0
        if self.pos_weight_max is not None:
            pos_weight = pos_weight.clamp(max=self.pos_weight_max)

        # NOTE: AbstractLoss는 nn.Module이 아니라 register_buffer()가 없음 (plain attribute로 보관)
        self.pos_weight = pos_weight

    def _weighted_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits, targets: (B, num_classes)
        """
        pos_weight = self.pos_weight.to(logits.device)

        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="mean")

        return loss

    def __call__(self, batch=None, model_output=None, **extras):
        logits  = model_output[self.logit_key]     # (B, num_classes)
        targets = batch[self.target_key].float()   # (B, num_classes)

        loss = self._weighted_bce_loss(logits, targets)

        logging_dict = OrderedDict({
            f"bce_loss_{self.logit_key}": loss.detach(),
            f"total_loss_{self.logit_key}": loss.detach(),
        })
        return loss, logging_dict

    @property
    def loss_keys(self):
        return {
            "target_label": self.target_key,
            "pred_label"  : self.logit_key,
        }


class MultiLabelFocalBCELoss(MultiLabelWeightedBCELoss):
    """`MultiLabelWeightedBCELoss` + focal 변조항. **gamma=0이면 부모와 비트 단위로 동일하다.**

    왜 부모를 상속했나: `pos_weight`(클래스별 neg/pos, 1.21~12.82)는 유지해야 한다. focal과
    pos_weight는 서로 다른 축을 고치는 항이라 둘 중 하나를 버릴 이유가 없다 — pos_weight는
    **클래스 간** 양/음성 비를 맞추고, focal은 **샘플 간** 쉬움/어려움을 재가중한다.
    ("focal + BCE"를 두 loss의 **합**으로 구현하지 않은 이유도 같다. 합치면 gamma 효과가
    BCE 항에 희석되고 조절할 가중치가 하나 더 늘어난다.)

    수식 (라벨별 독립, p = sigmoid(logit)):
        p_t   = p        (y=1) / 1-p      (y=0)
        w_t   = pos_weight (y=1) / 1       (y=0)      <- 부모와 동일
        loss  = mean( w_t * (1 - p_t)^gamma * BCE(logit, y) )

    gamma=2.0은 Lin et al. 2017의 값이다. **이 데이터에서 검증된 값이 아니다** —
    유병률이 6.9~45.2%로 focal이 설계된 극단 영역(<1%)과 멀고, 실측된 클래스별 AUROC는
    희귀할수록 **오히려 높다**(corr(prev, AUROC) = -0.283). 즉 이 loss는 "불균형이 병목"이라는
    가설의 **검정용**이지 개선이 확인된 설정이 아니다. 반드시 gamma=0 arm과 함께 돌릴 것.

    alpha는 일부러 넣지 않았다 — 원 논문의 alpha는 pos_weight와 같은 축(클래스 간 재가중)이라
    같이 쓰면 두 상수가 같은 것을 두 번 조절한다.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.gamma = float(kwargs.get("gamma", 2.0))
        assert self.gamma >= 0.0, f"gamma must be >= 0, got {self.gamma}"

    def _weighted_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight.to(logits.device)

        # reduction="none"으로 받아 원소별로 변조한 뒤 평균낸다. 부모는 여기서 바로
        # reduction="mean"을 쓴다 — gamma=0이면 아래 modulator가 정확히 1이라 결과가 같다.
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        if self.gamma == 0.0:
            return bce.mean()

        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        return ((1.0 - p_t).pow(self.gamma) * bce).mean()


class MultiLabelASLLoss(MultiLabelWeightedBCELoss):
    """Asymmetric Loss (Ridnik et al. 2021, ICCV) — 멀티라벨용 focal 확장.

    focal이 양성/음성에 **같은** gamma를 쓰는 것과 달리, ASL은 음성에 더 센 gamma를 주고
    (`gamma_neg` > `gamma_pos`), 확률이 `clip` 미만인 "확실한 음성"은 loss에서 아예 버린다
    (probability shifting). 멀티라벨에서 한 샘플의 음성 라벨이 양성보다 압도적으로 많다는
    구조를 겨냥한 것이고, 논문 기본값은 gamma_neg=4 / gamma_pos=0 / clip=0.05다.

        p_m   = max(p - clip, 0)                     (음성에만 적용)
        L_pos = y * (1-p)^gamma_pos * log(p)
        L_neg = (1-y) * p_m^gamma_neg * log(1-p_m)
        loss  = -mean( w_pos * L_pos + L_neg )

    ⚠️ 이 데이터에서 검증된 적 없다. 위 focal과 같은 이유로 **가설 검정용**이다. 그리고
    ASL은 원래 pos_weight 없이 쓰도록 설계됐으므로(gamma 비대칭이 그 역할을 한다),
    `use_pos_weight=False`가 기본이다 — True로 켜면 같은 축을 두 번 조절하게 된다.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.gamma_neg = float(kwargs.get("gamma_neg", 4.0))
        self.gamma_pos = float(kwargs.get("gamma_pos", 0.0))
        self.clip = float(kwargs.get("clip", 0.05))
        self.use_pos_weight = bool(kwargs.get("use_pos_weight", False))
        self.eps = float(kwargs.get("eps", 1e-8))

    def _weighted_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = (1.0 - p + self.clip).clamp(max=1.0) if self.clip > 0 else (1.0 - p)

        loss_pos = targets * torch.log(p.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps))

        if self.use_pos_weight:
            loss_pos = loss_pos * self.pos_weight.to(logits.device)

        # focal 항. detach: 원 구현(ASL 공식 저장소)이 gradient를 확률 변조항으로 흘리지
        # 않는다 — 흘리면 학습이 불안정해진다고 논문 4.3에 적혀 있다.
        with torch.no_grad():
            pt = p * targets + p_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            w = (1.0 - pt).pow(gamma)

        return -(w * (loss_pos + loss_neg)).mean()
