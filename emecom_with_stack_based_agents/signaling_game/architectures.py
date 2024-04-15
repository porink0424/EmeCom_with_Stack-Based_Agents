import dataclasses
from typing import cast

import torch
import torch.nn as nn
from torch.distributions import Categorical

from emecom_with_stack_based_agents.rl_spinn.rl_spinn import RlSpinnBase


class SenderEncoderBase(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        returns hidden state
        """
        raise NotImplementedError()


class ReceiverDecoderBase(nn.Module):
    def __init__(self):
        super().__init__()  # type: ignore

    def forward(
        self,
        hidden: torch.Tensor,
        hiddens: torch.Tensor,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        """
        returns logits
        """
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class SenderOutput:
    symbols: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ReceiverOutput:
    outputs: torch.Tensor
    transitions: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor


class Sender(nn.Module):
    def __init__(
        self,
        encoder: SenderEncoderBase,
        embed_dim: int,
        hidden_dim: int,
        max_length: int,
        vocab_size: int,
        force_eos: bool,
        device: torch.device,
    ):
        super().__init__()  # type: ignore
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.force_eos = force_eos
        self.device = device

        if force_eos:
            if self.max_length <= 1:
                raise ValueError(
                    "max_length must be greater than 1 when force_eos is True"
                )
            self.max_length = self.max_length - 1

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRUCell(embed_dim, hidden_dim)
        self.bos_embedding = nn.Parameter(torch.zeros(embed_dim).to(device))
        self.hidden_to_output = nn.Linear(hidden_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        nn.init.normal_(self.bos_embedding)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        hidden = self.encoder.forward(x)
        hidden = self.layer_norm.forward(hidden)
        input = (
            self.bos_embedding.unsqueeze(0)
            .expand(batch_size, self.embed_dim)
            .to(self.device)
        )

        symbols_history: list[torch.Tensor] = []
        log_probs_history: list[torch.Tensor] = []
        entropy_history: list[torch.Tensor] = []

        for _step in range(self.max_length):
            hidden = self.gru.forward(input, hidden)
            hidden = self.layer_norm.forward(hidden)

            logits = self.hidden_to_output.forward(hidden)
            distr = Categorical(logits=logits)
            entropy = cast(torch.Tensor, distr.entropy())
            if self.training:
                symbols = distr.sample()
            else:
                symbols = logits.argmax(dim=1)
            log_probs = cast(torch.Tensor, distr.log_prob(symbols))

            symbols_history.append(symbols)
            log_probs_history.append(log_probs)
            entropy_history.append(entropy)

            input = self.embedding.forward(symbols)

        symbols = torch.stack(symbols_history, dim=1).to(self.device)
        log_probs = torch.stack(log_probs_history, dim=1).to(self.device)
        entropy = torch.stack(entropy_history, dim=1).to(self.device)

        if self.force_eos:
            zeros = torch.zeros((batch_size, 1)).to(self.device)
            symbols = torch.cat([symbols, zeros.long()], dim=1).to(self.device)
            log_probs = torch.cat([log_probs, zeros], dim=1).to(self.device)
            entropy = torch.cat([entropy, zeros], dim=1).to(self.device)

        return SenderOutput(
            symbols,
            log_probs,
            entropy,
        )


class Receiver(nn.Module):
    def __init__(
        self,
        rl_spinn: RlSpinnBase,
        decoder: ReceiverDecoderBase,
    ):
        super().__init__()  # type: ignore

        self.rl_spinn = rl_spinn
        self.decoder = decoder

    def forward(self, messages: torch.Tensor):
        rl_spinn_output = self.rl_spinn.forward(messages)
        outputs = self.decoder.forward(
            rl_spinn_output.hidden,
            rl_spinn_output.hiddens,
            messages,
        )

        return ReceiverOutput(
            outputs,
            rl_spinn_output.transitions,
            rl_spinn_output.log_probs,
            rl_spinn_output.entropy,
        )
