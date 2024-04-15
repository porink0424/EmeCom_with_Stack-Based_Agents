import torch
import torch.nn as nn

from emecom_with_stack_based_agents.signaling_game.architectures import (
    ReceiverDecoderBase,
    SenderEncoderBase,
)


class AttValSenderEncoder(SenderEncoderBase):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        hidden_dim: int,
    ):
        super().__init__()  # type: ignore
        self.linear_layer = nn.Linear(n_attributes * n_values, hidden_dim)

    def forward(self, inputs: torch.Tensor):
        return self.linear_layer.forward(inputs)


class AttValReceiverDecoder(ReceiverDecoderBase):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        hidden_dim: int,
    ):
        super().__init__()  # type: ignore
        self.linear_layer = nn.Linear(hidden_dim, n_attributes * n_values)

    def forward(
        self,
        hidden: torch.Tensor,
        hiddens: torch.Tensor,
        messages: torch.Tensor,
    ):
        return self.linear_layer.forward(hidden)
