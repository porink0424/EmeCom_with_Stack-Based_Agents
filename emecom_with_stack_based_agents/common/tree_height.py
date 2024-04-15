import torch

from emecom_with_stack_based_agents.common.constants import REDUCE, SHIFT


class Node:
    def __init__(self, data: int):
        self.data = data
        self.left: Node | None = None
        self.right: Node | None = None


def calc_node_height(node: Node) -> int:
    if node.left is None and node.right is None:
        return 0
    elif node.left is not None and node.right is not None:
        return 1 + max(calc_node_height(node.left), calc_node_height(node.right))
    else:
        raise ValueError("Node has only one child")


# TODO: naive implementation, can it be optimized?
def calc_tree_heights(
    shift_reduce_sequences: torch.Tensor,
) -> torch.Tensor:
    batch_size = shift_reduce_sequences.size(0)

    heights = torch.zeros(batch_size)
    for i, sequence in enumerate(shift_reduce_sequences):
        nodes: list[Node] = []
        for j, action in enumerate(sequence):
            if action.item() == SHIFT:
                nodes.append(Node(j))
            elif action.item() == REDUCE:
                right = nodes.pop()
                left = nodes.pop()
                node = Node(j)
                node.left = left
                node.right = right
                nodes.append(node)
        root = nodes.pop()
        heights[i] = calc_node_height(root)
    return heights
