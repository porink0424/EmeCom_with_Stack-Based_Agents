import torch
import torch.nn.functional as F

from emecom_with_stack_based_agents.common.loss import DiffLossBase, DiffLossOutput


class AttValDiffLoss(DiffLossBase):
    def __init__(self, n_attributes: int, n_values: int):
        super().__init__()  # type: ignore
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(self, sender_inputs: torch.Tensor, receiver_outputs: torch.Tensor):
        batch_size = sender_inputs.size(0)
        reshaped_sender_inputs = sender_inputs.view(
            batch_size, self.n_attributes, self.n_values
        )
        reshaped_receiver_outputs = receiver_outputs.view(
            batch_size, self.n_attributes, self.n_values
        )

        acc = (
            (
                torch.sum(
                    (
                        reshaped_receiver_outputs.argmax(dim=2)
                        == reshaped_sender_inputs.argmax(dim=2)
                    ).detach(),
                    dim=1,
                )
                == self.n_attributes
            )
            .float()
            .mean()
        )
        acc_or = (
            (
                reshaped_receiver_outputs.argmax(dim=2)
                == reshaped_sender_inputs.argmax(dim=2)
            )
            .detach()
            .float()
            .mean()
        )

        cross_entropy_inputs = receiver_outputs.view(
            batch_size * self.n_attributes, self.n_values
        )
        cross_entropy_targets = reshaped_sender_inputs.argmax(dim=2).view(
            batch_size * self.n_attributes
        )
        loss = (
            F.cross_entropy(
                cross_entropy_inputs, cross_entropy_targets, reduction="none"
            )
            .view(batch_size, self.n_attributes)
            .mean(dim=1)
        )

        return DiffLossOutput(loss, acc, acc_or)
