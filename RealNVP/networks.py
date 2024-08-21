


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super().__init__()
        self.input_dim = input_dim
        self.mask = mask
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        x1 = x * self.mask
        x2 = x * (1 - self.mask)
        scale = self.scale_net(x1) * (1 - self.mask)
        translate = self.translate_net(x1) * (1 - self.mask)
        y1 = x1
        y2 = x2 * torch.exp(scale) + translate
        log_det = scale.sum(dim=1)
        return y1 + y2, log_det

    def inverse(self, y):
        y1 = y * self.mask
        y2 = y * (1 - self.mask)
        scale = self.scale_net(y1) * (1 - self.mask)
        translate = self.translate_net(y1) * (1 - self.mask)
        x1 = y1
        x2 = (y2 - translate) * torch.exp(-scale)
        return x1 + x2
    

class RealNVP(nn.Module):
    
    def __init__(self, image_size, input_dim, hidden_dim, num_layer, device):
        super().__init__()
        self.image_size = image_size
        self.input_dim = input_dim
        mean = torch.zeros(input_dim).to(device)
        cov = torch.eye(input_dim).to(device)
        self.prior = MultivariateNormal(mean, cov)
        self.device = device
        
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            mask = self.make_mask(i)
            self.layers.append(CouplingLayer(input_dim, hidden_dim, mask))
        self.to(device)
        
    def make_mask(self, layer_index):
        mask = torch.zeros(self.input_dim)
        if layer_index % 2==0: 
            mask[::2] = 1  
        else:
            mask[1::2] = 1
        mask = mask.to(self.device)
        return mask
        
    def forward(self, x):
        # from image x to Gaussian noise z; reshape image to 2 dimensional tensors
        x = x.view(-1, self.input_dim)
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_jacobian += log_det
        return x, log_det_jacobian

    def inverse(self, y):
        # from noise z to image x
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def log_prob(self, x):
        z, log_det_jacobian = self.forward(x)
        log_prob_z = self.prior.log_prob(z)
        return log_prob_z + log_det_jacobian
    
    def sampling(self, num_samples):
        z = self.prior.sample((num_samples,))
        x = self.inverse(z)
        x = x.view(-1, *self.image_size)
        return x
    

