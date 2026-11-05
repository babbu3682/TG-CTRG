import math
from torch.optim.lr_scheduler import LRScheduler


def _warmup_factor_epoch(t, warmup_epochs):
    """epoch 단위(`interval: epoch`) warmup 배율 — 기존 동작 그대로 유지.

    t가 정수이므로 (t+1)/(w+1)이 되어 첫 epoch도 lr 0 이 되지 않는다
    (원래 주석의 "breaking change" 의도). w=4면 0.2/0.4/0.6/0.8 → 1.0.
    """
    return (t + 1.0) / (warmup_epochs + 1.0)


def _warmup_factor_step(t, warmup_epochs):
    """step 단위(`interval: step`) warmup 배율 — MAE와 동일한 선형 램프 t/w.

    ⚠️ epoch용 식 (t+1)/(w+1)을 분수 t에 쓰면 안 된다: warmup_epochs=0.25 일 때
    t=0 에서 (0+1)/(0.25+1) = **0.8**, 즉 첫 step부터 base lr의 80%로 시작해
    warmup이 사실상 없어진다(검증 중 실제로 이 값이 나와서 잡았다). 분수 해상도로
    부를 때는 0에서 1로 선형으로 올리는 t/w 가 맞다 — t=0 인 단 한 step만 lr 0 이고,
    그건 warmup의 정의상 정상이다.
    """
    return min(max(t / max(warmup_epochs, 1e-12), 0.0), 1.0)


class CosineAnnealingWarmup(LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs,
        min_lr=0,
        last_epoch=-1,
        verbose=False,
        **extras,
    ):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def _decayed_lrs(self, t):
        progress = (t - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1e-12)
        progress = min(max(progress, 0.0), 1.0)
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * _warmup_factor_epoch(self.last_epoch, self.warmup_epochs) for base_lr in self.base_lrs]
        return self._decayed_lrs(self.last_epoch)

    def adjust_learning_rate(self, epoch_float):
        """`interval: step` 경로용. 분수 epoch을 받아 lr을 직접 세팅한다.

        ⚠️ 이 메서드는 원래 **존재하지 않았다**. `pillar/engines/{retrieval,classifier}.py`의
        train_one_epoch이 `lr_scheduler.adjust_learning_rate(batch_idx/len(dataloader) + epoch)`를
        호출하는데(MAE 관례), config에 `optimizer.scheduler.interval: step`을 주면
        scripts/train.py가 scheduler를 그대로 넘기므로 **첫 배치에서 AttributeError로 죽었다**.
        지금까지 모든 config가 `interval: epoch`이라 이 경로를 밟은 적이 없어 드러나지 않았다.

        epoch 단위 warmup은 해상도가 1 epoch이라, RA-RRG의 "50 warm-up steps"처럼
        1 epoch보다 짧은 warmup을 표현할 방법이 아예 없었다 — 그래서 이 저장소의
        retrieval config는 `warmup_epochs: 4`(= 9,544 step)를 쓰고 있었고, 실측 lr이
        base의 0.2/0.4/0.6/0.8배로 epoch 0~3을 소모했다. 이 메서드가 있으면
        `warmup_epochs`에 분수(예: 0.25)를 줄 수 있다.
        """
        if epoch_float < self.warmup_epochs:
            factor = _warmup_factor_step(epoch_float, self.warmup_epochs)
            lrs = [base_lr * factor for base_lr in self.base_lrs]
        else:
            lrs = self._decayed_lrs(epoch_float)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            # MAE와 동일하게 param group별 lr_scale(layer-wise lr decay)을 존중한다.
            group["lr"] = lr * group.get("lr_scale", 1.0)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        return lrs



class ConstantWarmup(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        min_lr=0,
        last_epoch=-1,
        verbose=False,
        **extras,
    ):
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def _decayed_lrs(self, t):
        return list(self.base_lrs)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * _warmup_factor_epoch(self.last_epoch, self.warmup_epochs) for base_lr in self.base_lrs]
        return self._decayed_lrs(self.last_epoch)

    def adjust_learning_rate(self, epoch_float):
        """`interval: step` 경로용. 분수 epoch을 받아 lr을 직접 세팅한다.

        ⚠️ 이 메서드는 원래 **존재하지 않았다**. `pillar/engines/{retrieval,classifier}.py`의
        train_one_epoch이 `lr_scheduler.adjust_learning_rate(batch_idx/len(dataloader) + epoch)`를
        호출하는데(MAE 관례), config에 `optimizer.scheduler.interval: step`을 주면
        scripts/train.py가 scheduler를 그대로 넘기므로 **첫 배치에서 AttributeError로 죽었다**.
        지금까지 모든 config가 `interval: epoch`이라 이 경로를 밟은 적이 없어 드러나지 않았다.

        epoch 단위 warmup은 해상도가 1 epoch이라, RA-RRG의 "50 warm-up steps"처럼
        1 epoch보다 짧은 warmup을 표현할 방법이 아예 없었다 — 그래서 이 저장소의
        retrieval config는 `warmup_epochs: 4`(= 9,544 step)를 쓰고 있었고, 실측 lr이
        base의 0.2/0.4/0.6/0.8배로 epoch 0~3을 소모했다. 이 메서드가 있으면
        `warmup_epochs`에 분수(예: 0.25)를 줄 수 있다.
        """
        if epoch_float < self.warmup_epochs:
            factor = _warmup_factor_step(epoch_float, self.warmup_epochs)
            lrs = [base_lr * factor for base_lr in self.base_lrs]
        else:
            lrs = self._decayed_lrs(epoch_float)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            # MAE와 동일하게 param group별 lr_scale(layer-wise lr decay)을 존중한다.
            group["lr"] = lr * group.get("lr_scale", 1.0)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        return lrs

