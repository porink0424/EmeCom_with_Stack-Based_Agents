import dataclasses
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from emecom_with_stack_based_agents.common.constants import REDUCE, SHIFT, SKIP
from emecom_with_stack_based_agents.common.length import find_length
from emecom_with_stack_based_agents.rl_spinn.tracking_lstm import TrackingLstm


@dataclasses.dataclass(frozen=True)
class RlSpinnOutput:
    hidden: torch.Tensor
    hiddens: torch.Tensor
    transitions: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor


class RlSpinnBase(nn.Module):
    def __init__(
        self,
        D_vec: int,
        D_tracking: int,
        max_length: int,
        vocab_size: int,
        child_sum_mode: bool,
        device: torch.device,
    ):
        super().__init__()  # type: ignore

    def forward(
        self,
        messages: torch.Tensor,
    ) -> RlSpinnOutput:
        raise NotImplementedError()


class RlSpinn(RlSpinnBase):
    def __init__(
        self,
        D_vec: int,
        D_tracking: int,
        max_length: int,
        vocab_size: int,
        child_sum_mode: bool,
        device: torch.device,
    ):
        super().__init__()  # type: ignore

        self.D_vec = D_vec
        self.D_tracking = D_tracking
        self.max_length = max_length
        self.device = device
        self.child_sum_mode = child_sum_mode

        self.tracking_lstm = TrackingLstm(D_vec, D_tracking)
        self.transition_net = nn.Linear(D_tracking, 2)
        self.embedding = nn.Embedding(
            vocab_size, 2 * D_vec
        )  # 2 * D_vec -> h_vec & c_vec
        self.composition_net = nn.Linear(
            D_vec + D_tracking if child_sum_mode else 2 * D_vec + D_tracking,
            5 * D_vec,
        )
        self.layer_norm = nn.LayerNorm(2 * D_vec, elementwise_affine=True)

    def forward(self, messages: torch.Tensor):
        batch_size = messages.size(0)

        queues = torch.full(size=(batch_size, self.max_length + 1), fill_value=-1).to(
            self.device
        )
        queue_indices = torch.full(size=(batch_size,), fill_value=-1).to(self.device)
        buffer_pointers = torch.full(size=(batch_size,), fill_value=-1).to(self.device)
        unexecuted_shifts: torch.Tensor = find_length(self.max_length, messages)
        thin_stacks = torch.zeros(
            batch_size, 2 * self.max_length - 1, 2 * self.D_vec
        ).to(
            self.device
        )  # 2 * D_vec -> h_vec & c_vec

        representations = self.embedding.forward(messages)

        transitions_history: list[torch.Tensor] = []
        log_probs_history: list[torch.Tensor] = []
        entropy_history: list[torch.Tensor] = []

        # Sure that the first two transitions are SHIFT (except 1-length messages).
        queues[:, 0] = 0
        queues[:, 1] = 1
        queue_indices[:] = 1
        buffer_pointers[:] = 2
        unexecuted_shifts = unexecuted_shifts - 2
        thin_stacks[:, 0] = self.layer_norm.forward(representations[:, 0])
        thin_stacks[:, 1] = self.layer_norm.forward(representations[:, 1])
        for _ in range(2):
            transitions = torch.full(size=(batch_size,), fill_value=SHIFT).to(
                self.device
            )
            log_probs = torch.full(size=(batch_size,), fill_value=0.0).to(self.device)
            entropy = torch.full(size=(batch_size,), fill_value=0.0).to(self.device)
            transitions_history.append(transitions)
            log_probs_history.append(log_probs)
            entropy_history.append(entropy)

        tracking_h = torch.zeros(batch_size, self.D_tracking).to(self.device)
        tracking_c = torch.zeros(batch_size, self.D_tracking).to(self.device)
        for t_step in range(2, 2 * self.max_length - 1):
            top_of_buffers = representations[torch.arange(batch_size), buffer_pointers][
                :, : self.D_vec
            ]
            top_of_stacks_1 = thin_stacks[
                torch.arange(batch_size),
                queues[torch.arange(batch_size), queue_indices],
            ][:, : self.D_vec]
            top_of_stacks_2 = thin_stacks[
                torch.arange(batch_size),
                queues[torch.arange(batch_size), queue_indices] - 1,
            ][:, : self.D_vec]

            tracking_h, tracking_c = self.tracking_lstm.forward(
                top_of_buffers,
                top_of_stacks_1,
                top_of_stacks_2,
                tracking_h,
                tracking_c,
            )

            transition_logits = self.transition_net.forward(tracking_h)

            # mask where reduce is not allowed
            reduce_mask = torch.zeros_like(transition_logits).to(self.device)
            reduce_mask[:, REDUCE] = (queue_indices <= 0).float() * torch.finfo(
                torch.float32
            ).min
            # mask where shift is not allowed
            shift_mask = torch.zeros_like(transition_logits).to(self.device)
            shift_mask[:, SHIFT] = (unexecuted_shifts == 0).float() * torch.finfo(
                torch.float32
            ).min
            transition_logits = transition_logits + reduce_mask + shift_mask

            distr = Categorical(logits=transition_logits)
            if self.training:
                transitions = distr.sample()
            else:
                transitions = transition_logits.argmax(dim=1)
            log_prob = cast(torch.Tensor, distr.log_prob(transitions))
            entropy = cast(torch.Tensor, distr.entropy())

            queue_indices = torch.clamp(
                (
                    queue_indices
                    + (transitions == SHIFT).long()
                    - (transitions == REDUCE).long()
                ),
                min=-1,
                max=self.max_length - 1,
            )
            buffer_pointers = torch.clamp(
                buffer_pointers + (transitions == SHIFT).long(),
                max=self.max_length - 1,
            )
            queues[torch.arange(batch_size), queue_indices] = t_step
            queues[torch.arange(batch_size), queue_indices + 1] = torch.where(
                transitions == REDUCE,
                -1,
                queues[torch.arange(batch_size), queue_indices + 1],
            ).to(self.device)
            unexecuted_shifts = unexecuted_shifts - (transitions == SHIFT).long()
            left = queues[torch.arange(batch_size), queue_indices]
            right = queues[torch.arange(batch_size), queue_indices + 1]
            stacks_1 = thin_stacks[torch.arange(batch_size), left]
            stacks_2 = thin_stacks[torch.arange(batch_size), right]

            # composition of stacks_1, stacks_2
            stacks_1_h, stacks_1_c = torch.chunk(stacks_1, 2, dim=1)
            stacks_2_h, stacks_2_c = torch.chunk(stacks_2, 2, dim=1)
            concat_h = (
                torch.cat([stacks_1_h + stacks_2_h, tracking_h], dim=1)
                if self.child_sum_mode
                else torch.cat([stacks_1_h, stacks_2_h, tracking_h], dim=1)
            )

            i, f_l, f_r, o, g = torch.chunk(
                self.composition_net.forward(concat_h), 5, dim=1
            )
            i = F.sigmoid(i)
            f_l = F.sigmoid(f_l)
            f_r = F.sigmoid(f_r)
            o = F.sigmoid(o)
            g = torch.tanh(g)

            c = f_l * stacks_2_c + f_r * stacks_1_c + i * g
            h = o * c

            thin_stacks[:, t_step] = self.layer_norm(
                torch.where(
                    transitions.unsqueeze(1) == REDUCE,
                    torch.cat((h, c), dim=1),
                    representations[torch.arange(batch_size), buffer_pointers],
                ).to(self.device)
            )

            transitions_history.append(transitions)
            log_probs_history.append(log_prob)
            entropy_history.append(entropy)

        transitions = torch.stack(transitions_history, dim=1)
        log_probs = torch.stack(log_probs_history, dim=1)
        entropy = torch.stack(entropy_history, dim=1)

        t_steps = (
            torch.arange(2 * self.max_length - 1).repeat(batch_size, 1).to(self.device)
        )
        messages_length: torch.Tensor = find_length(self.max_length, messages)
        transitions = torch.where(
            t_steps >= 2 * messages_length.unsqueeze(1) - 1,
            SKIP,
            transitions,
        )
        log_probs = torch.where(
            t_steps >= 2 * messages_length.unsqueeze(1) - 1,
            0.0,
            log_probs,
        )
        entropy = torch.where(
            t_steps >= 2 * messages_length.unsqueeze(1) - 1,
            0.0,
            entropy,
        )

        hidden = thin_stacks[
            torch.arange(batch_size).to(self.device), 2 * messages_length - 2
        ][:, : self.D_vec]
        hiddens = thin_stacks[:, : self.D_vec]

        return RlSpinnOutput(
            hidden,
            hiddens,
            transitions,
            log_probs,
            entropy,
        )
