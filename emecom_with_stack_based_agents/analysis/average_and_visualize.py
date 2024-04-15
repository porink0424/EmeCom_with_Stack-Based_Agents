# TODO: NEED TO BE REFACTORED.
import argparse
import os
import re
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np


# TODO: Highly dependent on the log format. Need to be more robust.
def construct_data(log_texts: list[str]) -> Any:
    results = []
    for log_text in log_texts:
        result = {}
        log_lines = log_text.split("\n")
        rl_spinn_results_start_index = 1
        rl_spinn_final_receiver_transitions_index = -1
        vanilla_results_start_index = -1
        vanilla_final_receiver_transitions_index = -1
        vanilla_right_results_start_index = -1
        vanilla_right_final_receiver_transitions_index = -1
        binary_results_start_index = -1
        binary_final_receiver_transitions_index = -1
        random_rl_spinn_results_start_index = -1
        random_rl_spinn_final_receiver_transitions_index = -1
        for i, line in enumerate(log_lines):
            if line == "***FINAL RECEIVER TRANSITIONS***":
                if rl_spinn_final_receiver_transitions_index == -1:
                    rl_spinn_final_receiver_transitions_index = i
                elif vanilla_final_receiver_transitions_index == -1:
                    vanilla_final_receiver_transitions_index = i
                elif vanilla_right_final_receiver_transitions_index == -1:
                    vanilla_right_final_receiver_transitions_index = i
                elif binary_final_receiver_transitions_index == -1:
                    binary_final_receiver_transitions_index = i
                else:
                    random_rl_spinn_final_receiver_transitions_index = i
            if line == "Vanilla Signaling Game":
                vanilla_results_start_index = i - 1
            if line == "Vanilla Right Signaling Game":
                vanilla_right_results_start_index = i - 1
            if line == "Binary Signaling Game":
                binary_results_start_index = i - 1
            if line == "Random RL-SPINN Signaling Game":
                random_rl_spinn_results_start_index = i - 1

        # get config
        config = {}
        config_line = re.sub(r" ", "", log_lines[0])
        config_line = re.sub(r"^Namespace\(", "", config_line)
        config_line = re.sub(r"\)$", "", config_line)
        for item in re.findall(r"(.*?)=(.*?),", config_line):
            config[item[0]] = item[1]
        config["device"] = re.search(
            r"device=device\(type=\'.*?\'\)", config_line
        ).group()[  # type: ignore
            7:
        ]
        result["config"] = config

        # get RL-SPINN results
        rl_spinn_train_results = []
        rl_spinn_eval_results = []
        for line in log_lines[rl_spinn_results_start_index:vanilla_results_start_index]:
            if line.startswith("***TRAIN***"):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                rl_spinn_train_results.append(dic)  # type: ignore
            elif line.startswith("***EVAL***") or line.startswith(
                "***INITIAL STATE EVAL***"
            ):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                rl_spinn_eval_results.append(dic)  # type: ignore
        transitions = []
        for line in log_lines[
            rl_spinn_final_receiver_transitions_index + 1 : vanilla_results_start_index
        ]:
            if line == "":
                continue
            transitions.append(list(line.split(" ")))  # type: ignore
        result["rl_spinn"] = {
            "train": rl_spinn_train_results,
            "eval": rl_spinn_eval_results,
            "transitions": transitions,
        }

        # get vanilla results
        vanilla_train_results = []
        vanilla_eval_results = []
        for line in log_lines[
            vanilla_results_start_index:vanilla_right_results_start_index
        ]:
            if line.startswith("***TRAIN***"):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                vanilla_train_results.append(dic)  # type: ignore
            elif line.startswith("***EVAL***") or line.startswith(
                "***INITIAL STATE EVAL***"
            ):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                vanilla_eval_results.append(dic)  # type: ignore
        transitions = []
        for line in log_lines[
            vanilla_final_receiver_transitions_index
            + 1 : vanilla_right_results_start_index
        ]:
            if line == "":
                continue
            transitions.append(list(line.split(" ")))  # type: ignore
        result["vanilla"] = {
            "train": vanilla_train_results,
            "eval": vanilla_eval_results,
            "transitions": transitions,
        }

        # get vanilla right results
        vanilla_right_train_results = []
        vanilla_right_eval_results = []
        for line in log_lines[
            vanilla_right_results_start_index:binary_results_start_index
        ]:
            if line.startswith("***TRAIN***"):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                vanilla_right_train_results.append(dic)  # type: ignore
            elif line.startswith("***EVAL***") or line.startswith(
                "***INITIAL STATE EVAL***"
            ):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                vanilla_right_eval_results.append(dic)  # type: ignore
        transitions = []
        for line in log_lines[
            vanilla_right_final_receiver_transitions_index
            + 1 : binary_results_start_index
        ]:
            if line == "":
                continue
            transitions.append(list(line.split(" ")))  # type: ignore
        result["vanilla_right"] = {
            "train": vanilla_right_train_results,
            "eval": vanilla_right_eval_results,
            "transitions": transitions,
        }

        # get binary results
        binary_train_results = []
        binary_eval_results = []
        for line in log_lines[
            binary_results_start_index:random_rl_spinn_results_start_index
        ]:
            if line.startswith("***TRAIN***"):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                binary_train_results.append(dic)  # type: ignore
            elif line.startswith("***EVAL***") or line.startswith(
                "***INITIAL STATE EVAL***"
            ):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                binary_eval_results.append(dic)  # type: ignore
        transitions = []
        for line in log_lines[
            binary_final_receiver_transitions_index
            + 1 : random_rl_spinn_results_start_index
        ]:
            if line == "":
                continue
            transitions.append(list(line.split(" ")))  # type: ignore
        result["binary"] = {
            "train": binary_train_results,
            "eval": binary_eval_results,
            "transitions": transitions,
        }

        # get Random RL-SPINN results
        random_rl_spinn_train_results = []
        random_rl_spinn_eval_results = []
        for line in log_lines[random_rl_spinn_results_start_index:]:
            if line.startswith("***TRAIN***"):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                random_rl_spinn_train_results.append(dic)  # type: ignore
            elif line.startswith("***EVAL***") or line.startswith(
                "***INITIAL STATE EVAL***"
            ):
                dic = {}
                for item in re.findall(r"([a-z].*?): (.*?)(,|$)", line):
                    dic[item[0]] = item[1]
                random_rl_spinn_eval_results.append(dic)  # type: ignore
        transitions = []
        for line in log_lines[random_rl_spinn_final_receiver_transitions_index + 1 :]:
            if line == "":
                continue
            transitions.append(list(line.split(" ")))  # type: ignore
        result["random_rl_spinn"] = {
            "train": random_rl_spinn_train_results,
            "eval": random_rl_spinn_eval_results,
            "transitions": transitions,
        }

        results.append(result)  # type: ignore
    return results  # type: ignore


def average_and_visualize_from_data(
    file_name: str,
    data: Any,
    train_or_eval: Literal["train", "eval"],
    target: str,
    title: str,
    xlabel: str,
    ylabel: str,
):
    n_epochs = int(data[0]["config"]["n_epochs"])
    for datum in data:
        if int(datum["config"]["n_epochs"]) != n_epochs:
            raise ValueError("n_epochs must be the same across data")
    if train_or_eval == "eval":
        n_epochs += 1

    rl_spinn_values_list: list[float] = [0.0 for _ in range(n_epochs)]
    rl_spinn_std_list: list[list[float]] = [[] for _ in range(n_epochs)]
    rl_spinn_values_count: list[int] = [0 for _ in range(n_epochs)]
    vanilla_values_list: list[float] = [0.0 for _ in range(n_epochs)]
    vanilla_std_list: list[list[float]] = [[] for _ in range(n_epochs)]
    vanilla_values_count: list[int] = [0 for _ in range(n_epochs)]
    vanilla_right_values_list: list[float] = [0.0 for _ in range(n_epochs)]
    vanilla_right_std_list: list[list[float]] = [[] for _ in range(n_epochs)]
    vanilla_right_values_count: list[int] = [0 for _ in range(n_epochs)]
    binary_values_list: list[float] = [0.0 for _ in range(n_epochs)]
    binary_std_list: list[list[float]] = [[] for _ in range(n_epochs)]
    binary_values_count: list[int] = [0 for _ in range(n_epochs)]
    random_rl_spinn_values_list: list[float] = [0.0 for _ in range(n_epochs)]
    random_rl_spinn_std_list: list[list[float]] = [[] for _ in range(n_epochs)]
    random_rl_spinn_values_count: list[int] = [0 for _ in range(n_epochs)]
    for datum in data:
        for i, dic in enumerate(datum["rl_spinn"][train_or_eval]):
            if not np.isnan(float(dic[target])):
                rl_spinn_values_list[i] += float(dic[target])
                rl_spinn_std_list[i].append(float(dic[target]))
                rl_spinn_values_count[i] += 1
        for i, dic in enumerate(datum["vanilla"][train_or_eval]):
            if not np.isnan(float(dic[target])):
                vanilla_values_list[i] += float(dic[target])
                vanilla_std_list[i].append(float(dic[target]))
                vanilla_values_count[i] += 1
        for i, dic in enumerate(datum["vanilla_right"][train_or_eval]):
            if not np.isnan(float(dic[target])):
                vanilla_right_values_list[i] += float(dic[target])
                vanilla_right_std_list[i].append(float(dic[target]))
                vanilla_right_values_count[i] += 1
        for i, dic in enumerate(datum["binary"][train_or_eval]):
            if not np.isnan(float(dic[target])):
                binary_values_list[i] += float(dic[target])
                binary_std_list[i].append(float(dic[target]))
                binary_values_count[i] += 1
        for i, dic in enumerate(datum["random_rl_spinn"][train_or_eval]):
            if not np.isnan(float(dic[target])):
                random_rl_spinn_values_list[i] += float(dic[target])
                random_rl_spinn_std_list[i].append(float(dic[target]))
                random_rl_spinn_values_count[i] += 1

    rl_spinn_values = np.array(
        [
            (
                rl_spinn_values_list[i] / rl_spinn_values_count[i]
                if rl_spinn_values_count[i] >= 1
                else np.nan
            )
            for i in range(n_epochs)
        ]
    )
    rl_spinn_std = np.array(
        [
            (
                np.std(rl_spinn_std_list[i], ddof=1) / np.sqrt(rl_spinn_values_count[i])
                if rl_spinn_values_count[i] >= 2
                else 0.0
            )
            for i in range(n_epochs)
        ]
    )
    vanilla_values = np.array(
        [
            (
                vanilla_values_list[i] / vanilla_values_count[i]
                if vanilla_values_count[i] >= 1
                else np.nan
            )
            for i in range(n_epochs)
        ]
    )
    vanilla_std = np.array(
        [
            (
                np.std(vanilla_std_list[i], ddof=1) / np.sqrt(vanilla_values_count[i])
                if vanilla_values_count[i] >= 2
                else 0.0
            )
            for i in range(n_epochs)
        ]
    )
    vanilla_right_values = np.array(
        [
            (
                vanilla_right_values_list[i] / vanilla_right_values_count[i]
                if vanilla_right_values_count[i] >= 1
                else np.nan
            )
            for i in range(n_epochs)
        ]
    )
    vanilla_right_std = np.array(
        [
            (
                np.std(vanilla_right_std_list[i], ddof=1)
                / np.sqrt(vanilla_right_values_count[i])
                if vanilla_right_values_count[i] >= 2
                else 0.0
            )
            for i in range(n_epochs)
        ]
    )
    binary_values = np.array(
        [
            (
                binary_values_list[i] / binary_values_count[i]
                if binary_values_count[i] >= 1
                else np.nan
            )
            for i in range(n_epochs)
        ]
    )
    binary_std = np.array(
        [
            (
                (np.std(binary_std_list[i], ddof=1) / np.sqrt(binary_values_count[i]))
                if binary_values_count[i] >= 2
                else 0.0
            )
            for i in range(n_epochs)
        ]
    )
    random_rl_spinn_values = np.array(
        [
            (
                random_rl_spinn_values_list[i] / random_rl_spinn_values_count[i]
                if random_rl_spinn_values_count[i] >= 1
                else np.nan
            )
            for i in range(n_epochs)
        ]
    )
    random_rl_spinn_std = np.array(
        [
            (
                np.std(random_rl_spinn_std_list[i], ddof=1)
                / np.sqrt(random_rl_spinn_values_count[i])
                if random_rl_spinn_values_count[i] >= 2
                else 0.0
            )
            for i in range(n_epochs)
        ]
    )

    fig, ax = plt.subplots()  # type: ignore
    ax.plot(  # type: ignore
        range(n_epochs),
        rl_spinn_values,
        label="RL-SPINN",
    )
    ax.fill_between(  # type: ignore
        range(n_epochs),
        rl_spinn_values + rl_spinn_std,  # type: ignore
        rl_spinn_values - rl_spinn_std,  # type: ignore
        alpha=0.25,
    )
    ax.plot(  # type: ignore
        range(n_epochs),
        vanilla_values,
        label="Vanilla",
    )
    ax.fill_between(  # type: ignore
        range(n_epochs),
        vanilla_values + vanilla_std,  # type: ignore
        vanilla_values - vanilla_std,  # type: ignore
        alpha=0.25,
    )
    ax.plot(  # type: ignore
        range(n_epochs),
        vanilla_right_values,
        label="Vanilla Right",
    )
    ax.fill_between(  # type: ignore
        range(n_epochs),
        vanilla_right_values + vanilla_right_std,  # type: ignore
        vanilla_right_values - vanilla_right_std,  # type: ignore
        alpha=0.25,
    )
    ax.plot(  # type: ignore
        range(n_epochs),
        binary_values,
        label="Binary",
    )
    ax.fill_between(  # type: ignore
        range(n_epochs),
        binary_values + binary_std,  # type: ignore
        binary_values - binary_std,  # type: ignore
        alpha=0.25,
    )
    ax.plot(  # type: ignore
        range(n_epochs),
        random_rl_spinn_values,
        label="Random",
    )
    ax.fill_between(  # type: ignore
        range(n_epochs),
        random_rl_spinn_values + random_rl_spinn_std,  # type: ignore
        random_rl_spinn_values - random_rl_spinn_std,  # type: ignore
        alpha=0.25,
    )
    ax.set_title(title)  # type: ignore
    ax.set_xlabel(xlabel)  # type: ignore
    ax.set_ylabel(ylabel)  # type: ignore
    ax.legend()  # type: ignore
    (
        os.mkdir(f"results_img/{dir_name}")
        if not os.path.isdir(f"results_img/{dir_name}")
        else None
    )
    fig.savefig(f"results_img/{dir_name}/{file_name}.png", facecolor="lightgray")  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", type=str)
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()
    dir_name = args.dir_name

    log_texts: list[str] = []
    for file_name in os.listdir(f"results_log/{dir_name}"):
        if not file_name.startswith(args.prefix):
            continue

        with open(f"results_log/{dir_name}/{file_name}", "r") as f:
            log_texts.append(f.read())
    data = construct_data(log_texts)

    targets = [
        "original loss",
        "acc",
        "acc_or",
        "messages length",
        "original sender entropy",
        "original receiver entropy",
    ]
    eval_targets = targets + [
        "tree heights",
    ]
    for target in targets:
        average_and_visualize_from_data(
            f"{dir_name}-train-{target}",
            data,
            "train",
            target,
            f"{dir_name}-train-{target}",
            "Epoch",
            target,
        )
    for target in eval_targets:
        average_and_visualize_from_data(
            f"{dir_name}-eval-{target}",
            data,
            "eval",
            target,
            f"{dir_name}-eval-{target}",
            "Epoch",
            target,
        )
