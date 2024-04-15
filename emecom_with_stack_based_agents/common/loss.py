import dataclasses

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class DiffLossOutput:
    loss: torch.Tensor
    acc: torch.Tensor
    acc_or: torch.Tensor


class DiffLossBase(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore

    def forward(
        self,
        sender_inputs: torch.Tensor,
        receiver_outputs: torch.Tensor,
    ) -> DiffLossOutput:
        raise NotImplementedError()
