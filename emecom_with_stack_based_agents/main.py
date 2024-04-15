import argparse
import random
from typing import Type

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from emecom_with_stack_based_agents.env import set_env
from emecom_with_stack_based_agents.rl_spinn.binary_branching_rl_spinn import (
    BinaryBranchingRlSpinn,
)
from emecom_with_stack_based_agents.rl_spinn.left_branching_rl_spinn import (
    LeftBranchingRlSpinn,
)
from emecom_with_stack_based_agents.rl_spinn.random_branching_rl_spinn import (
    RandomBranchingRlSpinn,
)
from emecom_with_stack_based_agents.rl_spinn.right_branching_rl_spinn import (
    RightBranchingRlSpinn,
)
from emecom_with_stack_based_agents.rl_spinn.rl_spinn import RlSpinn, RlSpinnBase
from emecom_with_stack_based_agents.signaling_game.architectures import Receiver, Sender
from emecom_with_stack_based_agents.signaling_game.attval.architectures import (
    AttValReceiverDecoder,
    AttValSenderEncoder,
)
from emecom_with_stack_based_agents.signaling_game.attval.data import (
    prepare_attval_data,
)
from emecom_with_stack_based_agents.signaling_game.attval.loss import AttValDiffLoss
from emecom_with_stack_based_agents.signaling_game.game import SenderReceiverReinforce
from emecom_with_stack_based_agents.signaling_game.trainer import GameTrainer


def rl_spinn_signaling_game(
    args: argparse.Namespace,
    rl_spinn: Type[RlSpinnBase],
    title: str = "RL-SPINN",
):
    print("*****", flush=True)
    print(f"{title} Signaling Game", flush=True)
    print("*****", flush=True)

    # prepare data
    train, validation = prepare_attval_data(
        args.n_attributes,
        args.n_values,
        args.seed,
        args.p_test,
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        sampler=RandomSampler(
            train,
            replacement=True,
            num_samples=args.batch_size,
        ),
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size)

    # prepare model
    sender = Sender(
        encoder=AttValSenderEncoder(
            n_attributes=args.n_attributes,
            n_values=args.n_values,
            hidden_dim=args.sender_hidden_dim,
        ),
        embed_dim=args.sender_emb_dim,
        hidden_dim=args.sender_hidden_dim,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        force_eos=args.force_eos,
        device=args.device,
    )
    receiver = Receiver(
        rl_spinn=rl_spinn(
            D_vec=args.receiver_hidden_dim,
            D_tracking=args.D_tracking,
            max_length=args.max_length,
            vocab_size=args.vocab_size,
            child_sum_mode=args.child_sum_mode,
            device=args.device,
        ),
        decoder=AttValReceiverDecoder(
            n_attributes=args.n_attributes,
            n_values=args.n_values,
            hidden_dim=args.receiver_hidden_dim,
        ),
    )
    loss = AttValDiffLoss(
        n_attributes=args.n_attributes,
        n_values=args.n_values,
    )
    game = SenderReceiverReinforce(
        sender=sender,
        receiver=receiver,
        loss=loss,
        sender_entropy_weight=args.sender_entropy_weight,
        receiver_entropy_weight=args.receiver_entropy_weight,
        length_pressure_weight=args.length_pressure_weight,
        max_length=args.max_length,
        device=args.device,
    )

    # prepare trainer
    trainer = GameTrainer(
        game=game,
        train_loader=train_loader,
        validation_loader=validation_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    # train
    return trainer.train_n_epochs(args.n_epochs)


def fix_seed(seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # type: ignore
        torch.cuda.manual_seed(seed)

        # NOTE: performance may be affected by these settings.
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
        torch.use_deterministic_algorithms(True)  # type: ignore


if __name__ == "__main__":
    set_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_attributes", type=int, required=True)
    parser.add_argument("--n_values", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--data_scale", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--D_tracking", type=int, default=300)
    parser.add_argument("--sender_emb_dim", type=int, default=5)
    parser.add_argument("--receiver_emb_dim", type=int, default=30)
    parser.add_argument("--sender_hidden_dim", type=int, default=500)
    parser.add_argument("--receiver_hidden_dim", type=int, default=500)
    parser.add_argument("--sender_entropy_weight", type=float, default=0.5)
    parser.add_argument("--receiver_entropy_weight", type=float, default=0.5)
    parser.add_argument("--length_pressure_weight", type=float, default=0.0)
    parser.add_argument("--force_eos", type=bool, default=False)
    parser.add_argument("--child_sum_mode", action="store_true")
    parser.add_argument("--rl_spinn_only", action="store_true")
    parser.add_argument("--vanilla_only", action="store_true")
    parser.add_argument("--p_test", type=float, default=0.1)
    parser.add_argument(
        "--device",
        type=str,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if args.rl_spinn_only and args.vanilla_only:
        raise ValueError("Cannot set both rl_spinn_only and vanilla_only to True.")
    elif args.rl_spinn_only:
        rl_spinn_signaling_game(args, RlSpinn, title="RL-SPINN")
    elif args.vanilla_only:
        rl_spinn_signaling_game(args, LeftBranchingRlSpinn, title="Vanilla")
    else:
        rl_spinn_signaling_game(args, RlSpinn, title="RL-SPINN")
        rl_spinn_signaling_game(args, LeftBranchingRlSpinn, title="Vanilla")
        rl_spinn_signaling_game(args, RightBranchingRlSpinn, title="Vanilla Right")
        rl_spinn_signaling_game(args, BinaryBranchingRlSpinn, title="Binary")
        rl_spinn_signaling_game(args, RandomBranchingRlSpinn, title="Random RL-SPINN")
