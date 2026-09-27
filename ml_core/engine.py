import torch
import torch.nn as nn

class AffineCouplingBlock(nn.Module):
    """
    Bijective coupling layer for invertible transformations.
    Ensures mathematical reversibility with zero data loss.
    """
    def __init__(self, channels):
        super(AffineCouplingBlock, self).__init__()
        self.half_channels = channels // 2
        
        self.net_s = nn.Sequential(
            nn.Conv2d(self.half_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, self.half_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.net_t = nn.Sequential(
            nn.Conv2d(self.half_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, self.half_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1, x2 = torch.split(x, self.half_channels, dim=1)
        s = self.net_s(x1)
        t = self.net_t(x1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = torch.split(y, self.half_channels, dim=1)
        s = self.net_s(y1)
        t = self.net_t(y1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)


class LosslessStegoINN(nn.Module):
    """
    Full Invertible Neural Network combining carrier and secret tensors.
    """
    def __init__(self, channels=6, num_blocks=4):
        super(LosslessStegoINN, self).__init__()
        self.blocks = nn.ModuleList([AffineCouplingBlock(channels) for _ in range(num_blocks)])

    def hide(self, carrier, secret):
        net_input = torch.cat([carrier, secret], dim=1)
        out = net_input
        for block in self.blocks:
            out = block(out)
        stego_carrier = out[:, :3, :, :]
        residual_latent = out[:, 3:, :, :]
        return stego_carrier, residual_latent

    def extract_and_restore(self, stego_carrier, residual_latent):
        out = torch.cat([stego_carrier, residual_latent], dim=1)
        for block in reversed(self.blocks):
            out = block.inverse(out)
        recovered_carrier = out[:, :3, :, :]
        recovered_secret = out[:, 3:, :, :]
        return recovered_carrier, recovered_secret