import torch
from torch.utils.data import Dataset


class AttValDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
        random_indices: list[int],
    ):
        super().__init__()  # type: ignore
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.random_indices = random_indices

    def __getitem__(self, index: int):
        assert index < len(self.random_indices)
        index = self.random_indices[index]
        one_hot = torch.zeros(self.n_attributes, self.n_values)
        for i in range(self.n_attributes):
            one_hot[i, index % self.n_values] = 1
            index //= self.n_values
        return one_hot.view(-1)

    def __len__(self):
        return len(self.random_indices)


def prepare_attval_data(n_attributes: int, n_values: int, seed: int, p_test: float):
    random_indices = torch.randperm(
        n_values**n_attributes, generator=torch.Generator().manual_seed(seed)
    )
    train_indices = random_indices[: int((1 - p_test) * len(random_indices))]
    test_indices = random_indices[int((1 - p_test) * len(random_indices)) :]
    train_dataset = AttValDataset(
        n_attributes, n_values, train_indices.tolist()  # type: ignore
    )
    test_dataset = AttValDataset(
        n_attributes, n_values, test_indices.tolist()  # type: ignore
    )
    return train_dataset, test_dataset
