from __future__ import annotations
from typing import Sequence, Callable
import logging
import torch.optim as optim
from torch.optim.lr_scheduler import (LambdaLR, MultiplicativeLR,
                                      StepLR, MultiStepLR,
                                      ExponentialLR, CosineAnnealingLR,
                                      ReduceLROnPlateau, CyclicLR, OneCycleLR,
                                      CosineAnnealingWarmRestarts)
from config import lr_lambda


class Scheduler:
    # https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    # https://sanghyu.tistory.com/113
    def __init__(self, optimizer: optim.optimizer, logger: logging.Logger):
        self.optimizer = optimizer
        self.logger = logger
        self.prev_lr = [group['lr'] for group in optimizer.param_groups]

    def LLR(
            self,
            factor: float = None,
            lr_lambda: Callable[[int], float] = lr_lambda,
            verbose=True
    ) -> optim.lr_scheduler:
        """LambdaLR

        lr_epoch = lr_init * lambda(epoch)
        lambda 표현식으로 작성한 함수를 통해 lr을 조절한다.
        초기 lr에 lambda 함수의 반환값을 곱해 lr을 계산한다.
        """
        return LambdaLR(
            self.optimizer,
            lambda epoch: factor ** epoch if factor is not None else lr_lambda,
            verbose
        )

    def MLR(
            self,
            factor: float = None,
            lr_lambda: Callable[[int], float] = lr_lambda,
            verbose=True
    ) -> optim.lr_scheduler:
        """MulticativeLR

        lr_epoch = lr_{epoch-1} * lambda(epoch)
        lambda 표현식으로 작성한 함수를 통해 lr을 조절한다.
        초기 lr에 lambda 함수의 반환값을 누적곱해 lr을 계산한다.
        """
        return MultiplicativeLR(
            self.optimizer,
            lambda epoch: factor ** epoch if factor is not None else lr_lambda,
            verbose
        )

    def SLR(self, step_size: int, gamma: float, verbose=True) \
            -> optim.lr_scheduler:
        """StepLR

        lr_epoch = gamma * lr_{epoch-1} if epoch % step_size == 0 else lr_{epoch-1}
        step_size의 epoch마다 gamma의 비율로 lr을 감소시킨다. (step_size마다 gamma를 곱한다)
        """
        return StepLR(
            self.optimizer,
            step_size, gamma,
            verbose
        )

    def MSLR(self, milestones: Sequence[int, ...], gamma: float, verbose=True) \
            -> optim.lr_scheduler:
        """MultiStepLR

        lr_epoch = gamma * lr_{epoch-1} if epoch in milestones else lr_{epoch-1}
        step_size 대신 lr을 감소시킬 epoch을 직접 지정한다.
        """
        return MultiStepLR(
            self.optimizer,
            milestones, gamma,
            verbose
        )

    def ELR(self, gamma: float, verbose=True) \
            -> optim.lr_scheduler:
        """ExponentialLR

        lr_epoch = gamma * lr_{epoch-1}
        lr이 지수함수를 따라 감소한다.
        """
        return ExponentialLR(
            self.optimizer,
            gamma,
            verbose
        )

    def CALR(self, T_max, eta_min, verbose=True) \
            -> optim.lr_scheduler:
        """CosineAnnealingLR

        lr이 eta_min까지 감소했다가 다시 초기 lr까지 올라온다.
        """
        return CosineAnnealingLR(
            self.optimizer,
            T_max, eta_min,
            verbose
        )

    def RLROP(
            self,
            # monitor: str,
            mode: str, threshold_mode: str = 'rel',
            factor: float = 0.1, patience: int = 10, threshold: float = 0.0,
            # cool_down: int = 0,
            min_lr: float | int = 0.0, eps: float = 1e-6,
            verbose=True
    ) -> optim.lr_scheduler:
        """ReduceLROnPlateau

        성능의 향상이 없을 때 lr을 감소시킨다.
        따라서 scheduler.step()에 mode에 따른 적절한 검증 성능을 넣어 주어야 한다. (loss나 score)
        # :param monitor:
        #     - 모니터링할 기준값
        #     - 원하는 val_metric 사용
        #     - default: 'val_loss' (검증 손실을 모니터링)
        :param mode:
            - 'min' or 'max' or 'auto'
            - 'min'은 성능이 감소할 때, 'max'는 성능이 증가할 때 lr을 감소시킴
            - 'auto'는 자동으로 모니터링되는 값에 따라 결정하는 옵션
        :param factor: lr 감소 비율 (new_lr = old_lr * factor)
        :param patience: 성능이 향상되지 않아도 lr을 감소시키지 않고 참을 epoch
        :param threshold: 성능 변화가 이 값을 초과해야만 성능 갱신으로 간주
        :param threshold_mode:
            - 성능 변화 측정 방법. 상대적(비율) 변화 혹은 절대적 변화
            - 'rel' or 'abs'
        # :param cool_down: lr 감소 후 이 값의 epoch 동안은 lr 미변경
        :param min_lr: lr의 최솟값 (이 아래로는 lr을 감소시키지 않음)
        :param eps: lr이 이 값보다 작아지면 더 이상 감소시키지 않음
        """
        return ReduceLROnPlateau(
            self.optimizer,
            mode, factor, patience, threshold, threshold_mode, min_lr, eps, verbose
        )

    def CLR(
            self,
            mode: str,
            base_lr: float, max_lr: float,
            step_size_up: int, step_size_down: int,
            gamma: float = 1,
            scale_fn: Callable = None,
            scale_mode: str = 'cycle',
            cycle_momentum: bool = True,
            base_momentum: float = 0.8,
            max_momentum: float = 0.9,
            verbose=True
    ) -> optim.lr_scheduler:
        """CycleLR

        학습 과정에서 명시적인 epoch 수나 iteration 수에 기반해 lr을 조절하지 않는다.
        대신 내부적으로 현재의 iteration을 추적해 이를 바탕으로 lr을 조절한다.
        따라서, CyclicLR의 .step()은 각 batch 처리 후에 호출되어야 한다.
        :param mode: 'triangular' or 'triangular2' or 'exp_range'
        :param base_lr: 최소 lr
        :param max_lr: 최대 lr
        :param step_size_up: cycle의 lr 증가 부분의 epoch
        :param step_size_down: cycle의 lr 감소 부분의 epoch (default: step_size_up)
        :param gamma:
            - 'exp_range' mode일 때의 주기 반복(iteration)마다 lr을 스케일링해 lr이 지수적으로 변화하도록 만듦
            - new_lr = gamma ** cycle 주기
        :param scale_fn: 커스텀 스케일링 함수 (이 옵션 사용 시 mode 무시)
        :param cycle_momentum: 주기적인 (lr cycle과 반대 방향으로) 모멘텀 조절 여부 (triangular 또는 triangular2 mode와 나이스)
        :param base_momentum: 최소 모멘텀
        :param max_momentum: 최대 모멘텀
        """
        return CyclicLR(
            self.optimizer,
            base_lr, max_lr, step_size_up, step_size_down, mode, gamma,
            scale_fn, scale_mode,
            cycle_momentum, base_momentum, max_momentum,
            verbose
        )

    def OCLR(
            self,
            max_lr: float,
            epochs: int,
            steps_per_epoch: int,
            pct_start: float = 0.1,
            anneal_strategy: str = 'cos',
            div_factor: int | float = 25,
            final_div_factor: int | float = 100,
            cycle_momentum: bool = True,
            base_momentum: float = 0.85,
            max_momentum: float = 0.95,
            verbose=True
    ) -> optim.lr_scheduler:
        """OneCycleLR

        초기 lr에서 한 cycle만 annealing한다.
        즉, 전체 epoch에 걸쳐 한 번의 학습 과정에서 lr 조절이 이루어지므로
        epoch가 아닌 iteration(step) 단위로 lr 조절이 이루어진다.
        이렇듯 학습 과정에서의 현재 단계(step)나 반복(iteration)에 따라 내부적으로 lr이 조절되므로,
        .step()은 일반적으로 각 batch 처리 후에 호출하며 호출 시 추가적인 값을 전달할 필요는 없다.
        iteration이란 학습 데이터를 모델에 넣어 학습해 가중치를 업데이트하는 한 번의 반복을 말한다. 즉, batch 크기와 같다.
        이러한 1-cycle 전략은 초기 lr에서 최대 lr까지 올라간 후 초기 lr보다 훨씬 낮은 lr로 annealing한다.

        :param max_lr: 최대 lr
        # :param total_steps:
        #     - cycle의 total steps
        #     - 'step'이란 학습 데이터를 모델에 넣어 학습해 가중치를 업데이트하는 한 번의 반복
        #     - total_steps = 총 epoch 수 * batch 크기
        :param epochs: 총 epoch
        :param steps_per_epoch: epoch당 학습하는 step 수 (batch_size)
        :param pct_start: lr의 증가 시점 (전체 주기에서의 상대적 비율)
        :param anneal_strategy:
            - 'cos' or 'linear'
            - lr 감소 전략
        :param div_factor: initial_lr = max_lr / div_factor
        :param final_div_factor: min_lr = initial_lr / final_div_factor
        :param cycle_momentum: 모멘텀 주기 사용 여부
        :param base_momentum: 모멘텀의 기본값
        :param max_momentum: 모멘텀의 최댓값
        """
        total_steps = epochs * steps_per_epoch
        return OneCycleLR(
            self.optimizer,
            max_lr, total_steps, epochs, steps_per_epoch, pct_start, anneal_strategy,
            cycle_momentum, base_momentum, max_momentum,
            div_factor, final_div_factor,
            verbose
        )

    def CAWR(self, T_0: int, T_mult: int, eta_min: float, verbose=True) \
            -> optim.lr_scheduler:
        """CosineAnnealingWarmRestarts

        Cosine annealing 함수를 따라 lr을 감소시키되 Ti epoch마다 lr을 초기화(재시작)한다.
        :param T_0: 첫 재시작을 위해 소요되는 epoch(iteration)
        :param T_mult: 재시작 후 Ti의 증가 factor
        :param eta_min: 최소 lr
        """
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0, T_mult, eta_min,
            verbose
        )

    def log_lr(self, scheduler: optim.lr_scheduler, epoch):
        current_lr = scheduler.get_last_lr()
        if any(prev != curr for prev, curr in zip(self.prev_lr, current_lr)):
            self.logger.info(
                f'Epoch {epoch}: Learning rate changed from {self.prev_lr} to {current_lr}'
            )
            self.prev_lr = current_lr

    @classmethod
    def get_scheduler(cls, name: str) -> optim.lr_scheduler:
        if hasattr(cls, name):
            scheduler = getattr(cls, name)
            return scheduler
        else:
            raise RuntimeError(f'Scheduler {name} not found.')
