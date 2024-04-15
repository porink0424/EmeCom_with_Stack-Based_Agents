import dataclasses

import torch
import torch.nn as nn

from emecom_with_stack_based_agents.common.baselines import MeanBaseline
from emecom_with_stack_based_agents.common.constants import SKIP
from emecom_with_stack_based_agents.common.length import find_length
from emecom_with_stack_based_agents.common.loss import DiffLossBase
from emecom_with_stack_based_agents.signaling_game.architectures import Receiver, Sender


@dataclasses.dataclass(frozen=True)
class GameOutput:
    optimized_loss: torch.Tensor
    original_loss: torch.Tensor
    acc: torch.Tensor
    acc_or: torch.Tensor
    original_sender_entropy: torch.Tensor
    original_receiver_entropy: torch.Tensor
    messages_length_mean: torch.Tensor
    messages: torch.Tensor
    receiver_transitions: torch.Tensor


class GameBase(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore

    def forward(
        self,
        sender_inputs: torch.Tensor,
    ) -> GameOutput:
        raise NotImplementedError()


class SenderReceiverReinforce(GameBase):
    def __init__(
        self,
        sender: Sender,
        receiver: Receiver,
        loss: DiffLossBase,
        sender_entropy_weight: float,
        receiver_entropy_weight: float,
        length_pressure_weight: float,
        max_length: int,
        device: torch.device,
    ):
        super().__init__()  # type: ignore
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.sender_entropy_weight = sender_entropy_weight
        self.receiver_entropy_weight = receiver_entropy_weight
        self.length_pressure_weight = length_pressure_weight
        self.max_length = max_length
        self.device = device

        self.baseline = MeanBaseline()

    def forward(self, sender_inputs: torch.Tensor):
        sender_output = self.sender.forward(sender_inputs)
        receiver_output = self.receiver.forward(sender_output.symbols)
        loss_output = self.loss.forward(sender_inputs, receiver_output.outputs)
        original_loss = loss_output.loss.mean()

        messages_length = find_length(self.max_length, sender_output.symbols)
        messages_length_mean = messages_length.float().mean()

        not_eosed = (
            torch.arange(self.max_length)
            .repeat(sender_inputs.size(0), 1)
            .to(self.device)
            < messages_length.unsqueeze(1)
        ).float()
        not_eosed_sender_entropy = (
            torch.sum(sender_output.entropy * not_eosed, dim=1)
        ) / messages_length
        original_sender_entropy = not_eosed_sender_entropy.mean()
        weighted_entropy = original_sender_entropy * self.sender_entropy_weight

        not_padding_receiver_entropy = torch.sum(receiver_output.entropy, dim=1)
        shift_or_reduce_count = torch.sum(receiver_output.transitions != SKIP, dim=1)
        not_padding_receiver_entropy = (
            not_padding_receiver_entropy / shift_or_reduce_count
        )
        original_receiver_entropy = not_padding_receiver_entropy.mean()
        weighted_entropy = weighted_entropy + (
            original_receiver_entropy * self.receiver_entropy_weight
        )

        log_probs = sender_output.log_probs.sum(dim=1) + receiver_output.log_probs.sum(
            dim=1
        )

        length_pressure = messages_length * self.length_pressure_weight
        optimized_loss = (
            loss_output.loss.mean()
            + length_pressure.mean()
            + (
                (
                    loss_output.loss.detach()
                    + length_pressure.detach()
                    - self.baseline.predict(
                        loss_output.loss.detach() + length_pressure.detach()
                    )
                )
                * log_probs
            ).mean()
        )

        optimized_loss = optimized_loss - weighted_entropy

        if self.training:
            self.baseline.update(loss_output.loss + length_pressure)

        return GameOutput(
            optimized_loss,
            original_loss,
            loss_output.acc,
            loss_output.acc_or,
            original_sender_entropy,
            original_receiver_entropy,
            messages_length_mean,
            sender_output.symbols,
            receiver_output.transitions,
        )
