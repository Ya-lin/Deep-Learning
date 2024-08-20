


import torch
import torch.nn as nn


class CheckerboardCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        mask = self.checkerboard_mask(x.size())
        x1 = x * mask
        x2 = x * (1 - mask)
        scale = self.scale_net(x1) * (1 - mask)
        translate = self.translate_net(x1) * (1 - mask)
        y1 = x1
        y2 = x2 * torch.exp(scale) + translate
        return y1 + y2, scale

    def inverse(self, y):
        mask = self.checkerboard_mask(y.size())
        y1 = y * mask
        y2 = y * (1 - mask)
        scale = self.scale_net(y1) * (1 - mask)
        translate = self.translate_net(y1) * (1 - mask)
        x1 = y1
        x2 = (y2 - translate) * torch.exp(-scale)
        return x1 + x2

    def checkerboard_mask(self, size):
        mask = torch.zeros(size)
        mask[:, ::2] = 1
        mask[:, 1::2] = 0
        return mask
    

class RealNVP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(CheckerboardCouplingLayer(input_dim, hidden_dim))

    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.layers:
            x, scale = layer(x)
            log_det_jacobian += scale.sum(dim=1)
        return x, log_det_jacobian

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def log_prob(self, x):
        z, log_det_jacobian = self.forward(x)
        log_prob_z = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.size(1) * torch.log(torch.tensor(2 * torch.pi))
        return log_prob_z + log_det_jacobian
    
    