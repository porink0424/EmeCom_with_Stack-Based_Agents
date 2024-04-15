import torch


class MeanBaseline:
    def __init__(self):
        self.mean = torch.tensor(0.0, requires_grad=False)
        self.n = 0

    def update(self, loss: torch.Tensor) -> None:
        self.n += 1
        if self.mean.device != loss.device:
            self.mean = self.mean.to(loss.device)
        self.mean += (loss.detach().mean().item() - self.mean) / self.n

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mean.device != loss.device:
            self.mean = self.mean.to(loss.device)
        return self.mean
