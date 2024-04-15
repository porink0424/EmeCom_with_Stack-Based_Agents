import dataclasses
from typing import cast

import torch
from torch.utils.data import DataLoader

from emecom_with_stack_based_agents.common.constants import REDUCE, SHIFT
from emecom_with_stack_based_agents.common.tree_height import calc_tree_heights
from emecom_with_stack_based_agents.signaling_game.game import GameBase


@dataclasses.dataclass()
class EpochResult:
    optimized_loss: float
    original_loss: float
    acc: float
    acc_or: float
    original_sender_entropy: float
    original_receiver_entropy: float
    messages_length_mean: float


class GameTrainer:
    def __init__(
        self,
        game: GameBase,
        train_loader: DataLoader[torch.Tensor],
        validation_loader: DataLoader[torch.Tensor],
        lr: float,
        weight_decay: float,
        device: torch.device,
    ):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device

        self.game = game
        self.game.to(device)
        self.optimizer = torch.optim.Adam(
            self.game.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.train_acc_max = -1.0
        self.train_acc_or_max = -1.0
        self.test_acc_max = -1.0
        self.test_acc_or_max = -1.0

        self.train_acc_max_receiver_transitions = None
        self.train_acc_max_receiver_messages = None
        self.train_acc_or_max_receiver_transitions = None
        self.train_acc_or_max_receiver_messages = None
        self.test_acc_max_receiver_transitions = None
        self.test_acc_max_receiver_messages = None
        self.test_acc_or_max_receiver_transitions = None
        self.test_acc_or_max_receiver_messages = None
        self.last_receiver_transitions = None
        self.last_receiver_messages = None

    def train_one_epoch(self, epoch_index: int):
        self.game.train()

        running_epoch_result = EpochResult(
            optimized_loss=0.0,
            original_loss=0.0,
            acc=0.0,
            acc_or=0.0,
            original_sender_entropy=0.0,
            original_receiver_entropy=0.0,
            messages_length_mean=0.0,
        )
        count = 0

        for _i, data in enumerate(self.train_loader):
            data: torch.Tensor = data.to(self.device)

            self.optimizer.zero_grad()
            game_output = self.game.forward(data)
            game_output.optimized_loss.backward()  # type: ignore
            self.optimizer.step()

            running_epoch_result.optimized_loss += (
                game_output.optimized_loss.item() * data.size(0)
            )
            running_epoch_result.original_loss += (
                game_output.original_loss.item() * data.size(0)
            )
            running_epoch_result.acc += game_output.acc.item() * data.size(0)
            running_epoch_result.acc_or += game_output.acc_or.item() * data.size(0)
            running_epoch_result.original_sender_entropy += (
                game_output.original_sender_entropy.item() * data.size(0)
            )
            running_epoch_result.original_receiver_entropy += (
                game_output.original_receiver_entropy.item()
            ) * data.size(0)
            running_epoch_result.messages_length_mean += (
                game_output.messages_length_mean.item() * data.size(0)
            )
            count += data.size(0)

        return EpochResult(
            optimized_loss=running_epoch_result.optimized_loss / count,
            original_loss=running_epoch_result.original_loss / count,
            acc=running_epoch_result.acc / count,
            acc_or=running_epoch_result.acc_or / count,
            original_sender_entropy=running_epoch_result.original_sender_entropy
            / count,
            original_receiver_entropy=running_epoch_result.original_receiver_entropy
            / count,
            messages_length_mean=running_epoch_result.messages_length_mean / count,
        )

    def eval_one_epoch(self, epoch_index: int):
        self.game.eval()

        with torch.no_grad():
            running_epoch_result = EpochResult(
                optimized_loss=0.0,
                original_loss=0.0,
                acc=0.0,
                acc_or=0.0,
                original_sender_entropy=0.0,
                original_receiver_entropy=0.0,
                messages_length_mean=0.0,
            )
            count = 0
            messages: list[torch.Tensor] = []
            epoch_receiver_transitions: list[torch.Tensor] = []

            for _i, data in enumerate(self.validation_loader):
                data: torch.Tensor = data.to(self.device)

                game_output = self.game.forward(data)

                running_epoch_result.optimized_loss += (
                    game_output.optimized_loss.item() * data.size(0)
                )
                running_epoch_result.original_loss += (
                    game_output.original_loss.item() * data.size(0)
                )
                running_epoch_result.acc += game_output.acc.item() * data.size(0)
                running_epoch_result.acc_or += game_output.acc_or.item() * data.size(0)
                running_epoch_result.original_sender_entropy += (
                    game_output.original_sender_entropy.item() * data.size(0)
                )
                running_epoch_result.original_receiver_entropy += (
                    game_output.original_receiver_entropy.item() * data.size(0)
                )
                running_epoch_result.messages_length_mean += (
                    game_output.messages_length_mean.item() * data.size(0)
                )
                count += data.size(0)
                epoch_receiver_transitions.append(game_output.receiver_transitions)
                messages.append(game_output.messages)

        return (
            EpochResult(
                optimized_loss=running_epoch_result.optimized_loss / count,
                original_loss=running_epoch_result.original_loss / count,
                acc=running_epoch_result.acc / count,
                acc_or=running_epoch_result.acc_or / count,
                original_sender_entropy=running_epoch_result.original_sender_entropy
                / count,
                original_receiver_entropy=running_epoch_result.original_receiver_entropy
                / count,
                messages_length_mean=running_epoch_result.messages_length_mean / count,
            ),
            torch.cat(epoch_receiver_transitions),
            torch.cat(messages),
        )

    def train_n_epochs(self, n_epochs: int):
        original_loss_data: list[float] = []

        # first eval
        epoch_result, transitions, messages = self.eval_one_epoch(epoch_index=0)
        tree_heights = calc_tree_heights(transitions)
        print(
            (
                f"***INITIAL STATE EVAL***  loss: {epoch_result.optimized_loss:.5f}, original loss: {epoch_result.original_loss:.5f}, "
                f"acc: {epoch_result.acc:.5f}, acc_or: {epoch_result.acc_or:.5f}, original sender entropy: {epoch_result.original_sender_entropy:.5f}, "
                f"original receiver entropy: {epoch_result.original_receiver_entropy:.5f}, messages length: {epoch_result.messages_length_mean:.5f}, tree heights: {tree_heights.mean().item():.5f}"
            ),
            flush=True,
        )

        for epoch_index in range(n_epochs):
            print(f"EPOCH {epoch_index + 1}:", flush=True)

            train_epoch_result = self.train_one_epoch(epoch_index)
            print(
                (
                    f"***TRAIN*** loss: {train_epoch_result.optimized_loss:.5f}, original loss: {train_epoch_result.original_loss:.5f}, "
                    f"acc: {train_epoch_result.acc:.5f}, acc_or: {train_epoch_result.acc_or:.5f}, original sender entropy: {train_epoch_result.original_sender_entropy:.5f}, "
                    f"original receiver entropy: {train_epoch_result.original_receiver_entropy:.5f}, messages length: {train_epoch_result.messages_length_mean:.5f}"
                ),
                flush=True,
            )
            original_loss_data.append(train_epoch_result.original_loss)

            epoch_result, transitions, messages = self.eval_one_epoch(epoch_index)

            if train_epoch_result.acc > self.train_acc_max:
                self.train_acc_max = train_epoch_result.acc
                self.train_acc_max_receiver_transitions = transitions
                self.train_acc_max_receiver_messages = messages
            if train_epoch_result.acc_or > self.train_acc_or_max:
                self.train_acc_or_max = train_epoch_result.acc_or
                self.train_acc_or_max_receiver_transitions = transitions
                self.train_acc_or_max_receiver_messages = messages
            if epoch_result.acc > self.test_acc_max:
                self.test_acc_max = epoch_result.acc
                self.test_acc_max_receiver_transitions = transitions
                self.test_acc_max_receiver_messages = messages
            if epoch_result.acc_or > self.test_acc_or_max:
                self.test_acc_or_max = epoch_result.acc_or
                self.test_acc_or_max_receiver_transitions = transitions
                self.test_acc_or_max_receiver_messages = messages
            self.last_receiver_transitions = transitions
            self.last_receiver_messages = messages

            tree_heights = calc_tree_heights(transitions)
            print(
                (
                    f"***EVAL***  loss: {epoch_result.optimized_loss:.5f}, original loss: {epoch_result.original_loss:.5f}, "
                    f"acc: {epoch_result.acc:.5f}, acc_or: {epoch_result.acc_or:.5f}, original sender entropy: {epoch_result.original_sender_entropy:.5f}, "
                    f"original receiver entropy: {epoch_result.original_receiver_entropy:.5f}, messages length: {epoch_result.messages_length_mean:.5f}, tree heights: {tree_heights.mean().item():.5f}"
                ),
                flush=True,
            )

        torch.set_printoptions(edgeitems=1000)  # type: ignore
        print(
            "***FINAL RECEIVER TRANSITIONS***",
            flush=True,
        )
        for head_title, transitions_list, messages_list in [
            (
                "train acc max",
                self.train_acc_max_receiver_transitions,
                self.train_acc_max_receiver_messages,
            ),
            (
                "train acc_or max",
                self.train_acc_or_max_receiver_transitions,
                self.train_acc_or_max_receiver_messages,
            ),
            (
                "test acc max",
                self.test_acc_max_receiver_transitions,
                self.test_acc_max_receiver_messages,
            ),
            (
                "test acc_or max",
                self.test_acc_or_max_receiver_transitions,
                self.test_acc_or_max_receiver_messages,
            ),
            (
                "last epoch",
                self.last_receiver_transitions,
                self.last_receiver_messages,
            ),
        ]:
            print(
                f"**{head_title}**",
                flush=True,
            )
            for transitions, messages in zip(
                cast(torch.Tensor, transitions_list),
                cast(torch.Tensor, messages_list),
            ):
                print(
                    *[
                        (
                            "SHIFT"
                            if action.item() == SHIFT
                            else "REDUCE" if action.item() == REDUCE else "SKIP"
                        )
                        for action in transitions
                    ],
                    flush=True,
                )
                net_message: list[str] = []
                for symbol in messages.tolist():  # type: ignore
                    symbol = str(symbol)  # type: ignore
                    net_message.append(symbol)
                    if symbol == "0":
                        break
                print(
                    "-".join(net_message),
                    flush=True,
                )

        return original_loss_data
