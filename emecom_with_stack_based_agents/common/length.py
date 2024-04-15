import torch


def find_length(
    max_length: int,
    messages: torch.Tensor,
) -> torch.Tensor:
    """
    example
    [1,2,3,0,0] -> length = 4
    [0,0,0,0,0] -> length = 1
    [1,2,3,4,5] -> length = 5
    [1,2,0,0,1] -> length = 3
    """
    zero_mask = messages == 0
    lengths = max_length - torch.sum(zero_mask.cumsum(dim=1) > 0, dim=1)
    lengths = torch.clamp(lengths + 1, max=max_length)

    return lengths
