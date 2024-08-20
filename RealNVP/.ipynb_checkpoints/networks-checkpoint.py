

import torch
import torch.nn as nn

class CheckerboardCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CheckerboardCouplingLayer, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )

    def forward(self, x):
        mask = self.create_checkerboard_mask(x.size())
        x1 = x * mask
        x2 = x * (1 - mask)
        scale = self.scale_net(x1)
        translate = self.translate_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(scale) + translate
        return y1 + y2, scale

    def inverse(self, y):
        mask = self.create_checkerboard_mask(y.size())
        y1 = y * mask
        y2 = y * (1 - mask)
        scale = self.scale_net(y1)
        translate = self.translate_net(y1)
        x1 = y1
        x2 = (y2 - translate) * torch.exp(-scale)
        return x1 + x2

    def create_checkerboard_mask(self, size):
        mask = torch.zeros(size)
        mask[:, ::2] = 1
        mask[:, 1::2] = 0
        return mask


