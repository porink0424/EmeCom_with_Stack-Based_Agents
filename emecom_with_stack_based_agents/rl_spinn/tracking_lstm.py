import torch
import torch.nn as nn


class TrackingLstm(nn.Module):
    def __init__(self, D_vec: int, D_tracking: int):
        super().__init__()  # type: ignore
        self.D_vec = D_vec
        self.D_traking = D_tracking

        self.lstm = nn.LSTMCell(input_size=3 * self.D_vec, hidden_size=self.D_traking)
        self.layer_norm = nn.LayerNorm(self.D_traking, elementwise_affine=True)

    def forward(
        self,
        buffer: torch.Tensor,
        stack_1: torch.Tensor,
        stack_2: torch.Tensor,
        tracking_h: torch.Tensor,
        tracking_c: torch.Tensor,
    ):
        new_tracking_h, new_tracking_c = self.lstm.forward(
            torch.cat([buffer, stack_1, stack_2], dim=1),
            (tracking_h, tracking_c),
        )
        new_tracking_h = self.layer_norm.forward(new_tracking_h)

        return new_tracking_h, new_tracking_c
