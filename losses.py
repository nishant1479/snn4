import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseTemporalLoss(nn.Module, ABC):
    """
    Base class for temporal losses.

    Total Loss:
        (1-lambda) * Temporal CE
        + lambda * MSE Regularization
    """

    def __init__(
        self,
        criterion: nn.Module,
        means: float = 1.0,
        lamb: float = 0.0,
    ):
        super().__init__()

        self.criterion = criterion
        self.means = means
        self.lamb = lamb
        self.mse = nn.MSELoss()

    @abstractmethod
    def temporal_weights(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns normalized temporal weights.

        Shape:
            (T,)
        """
        pass

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        outputs : [B,T,C]
        labels  : [B]

        Returns
        -------
        total_loss
        temporal_weights
        """

        T = outputs.size(1)

        weights = self.temporal_weights(
            T,
            outputs.device,
            outputs.dtype,
        )

        temporal_loss = 0.0

        for t in range(T):
            ce = self.criterion(outputs[:, t, ...], labels)
            temporal_loss = temporal_loss + weights[t] * ce

        if self.lamb > 0:

            target = torch.full_like(outputs, self.means)

            mse_loss = self.mse(outputs, target)

        else:

            mse_loss = outputs.new_tensor(0.0)

        total_loss = (1.0 - self.lamb) * temporal_loss + self.lamb * mse_loss

        return total_loss, weights


class OriginalTETLoss(BaseTemporalLoss):
    """
    Original TET

    w_t = 1/T
    """

    def temporal_weights(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        return torch.full(
            (T,),
            1.0 / T,
            device=device,
            dtype=dtype,
        )


class LinearWeightedTETLoss(BaseTemporalLoss):
    """
    Linear Weighted TET

    w_t = t / sum(i)

    Example

    T=5

    [1,2,3,4,5]/15
    """

    def temporal_weights(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        weights = torch.arange(
            1,
            T + 1,
            device=device,
            dtype=dtype,
        )

        weights = weights / weights.sum()

        return weights


def build_temporal_loss(
    loss_name: str,
    criterion: nn.Module,
    means: float = 1.0,
    lamb: float = 0.0,
) -> BaseTemporalLoss:
    """
    Factory function.

    Easily extensible to

    - exponential
    - entropy
    - learnable
    """

    loss_name = loss_name.lower()

    if loss_name == "tet":
        return OriginalTETLoss(
            criterion=criterion,
            means=means,
            lamb=lamb,
        )

    elif loss_name == "linear":
        return LinearWeightedTETLoss(
            criterion=criterion,
            means=means,
            lamb=lamb,
        )

    raise ValueError(
        f"Unknown temporal loss '{loss_name}'"
    )