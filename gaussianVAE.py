import torch
from torch import nn


class Encoder(torch.nn.Module):
    def __init__(self, N, input_shape, skip=True, norm="layer", dense=False):
        super().__init__()
        input_channel = 2 if dense else 17
        self.N = N
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64) if norm == "batch" else nn.LayerNorm([64, 5, 5]),
            nn.SiLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128) if norm == "batch" else nn.LayerNorm([128, 3, 3]),
            nn.SiLU(),
        )
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Sequential(
            nn.Linear(3 * 3 * 128, N),
            nn.BatchNorm2d(N) if norm == "batch" else nn.LayerNorm([N]),
            nn.SiLU(),
        )
        self.fc_log_var = nn.Sequential(
            nn.Linear(3 * 3 * 128, N),
            nn.BatchNorm2d(N) if norm == "batch" else nn.LayerNorm([N]),
            nn.SiLU(),
        )

        self.skip = skip
        if self.skip:
            self.resample = nn.Sequential(
                nn.Conv2d(17, 128, 1, stride=4, padding=0),
                nn.BatchNorm2d(128) if norm == "batch" else nn.LayerNorm([128, 3, 3]),
            )

    def forward(self, x):
        if self.skip:
            resample = self.resample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.skip:
            x = x + resample
        x = self.flatten(x)

        mean = self.fc_mean(x)
        mean = mean.reshape(-1, self.N)

        EPS_STD = 1e-6
        log_var = self.fc_log_var(x) + EPS_STD

        return mean, log_var


class Decoder(torch.nn.Module):
    output_shape: torch.Size
    N: int  # number of categorical distributions
    K: int  # number of classes

    def __init__(
        self,
        N: int,
        output_shape,
        skip=True,
        norm="layer",
    ):
        super().__init__()
        self.N = N
        self.output_shape = output_shape

        # First main block (Transpose Convolutions)
        self.fc1 = nn.Linear(N, 128 * 3 * 3)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0),
            nn.BatchNorm2d(64) if norm == "batch" else nn.LayerNorm([64, 5, 5]),
            nn.SiLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 17, 4, stride=2, padding=1),
            nn.BatchNorm2d(17) if norm == "batch" else nn.LayerNorm([17, 10, 10]),
            nn.SiLU(),
        )

        self.skip = skip
        if self.skip:
            self.resample = nn.Sequential(
                nn.ConvTranspose2d(128, 17, 1, stride=4, output_padding=1),
                nn.BatchNorm2d(17) if norm == "batch" else nn.LayerNorm([17, 10, 10]),
            )

        self.fc2 = nn.Linear(17 * 10 * 10, 17 * 10 * 10)
        self.fc2.bias.data.fill_(-3)

    def forward(self, z):
        # First main block processing
        x = self.fc1(z.reshape(-1, self.N)).reshape(-1, 128, 3, 3)
        if self.skip:
            resample = self.resample(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.skip:
            x = x + resample
        x = self.fc2((x).view(-1, 17 * 10 * 10))
        x_hat = x.view((-1,) + self.output_shape)

        return x_hat


def gaussian_sample(mu, log_var):
    # mu: bs, N
    # std: bs, N
    std = torch.exp(0.5 * log_var)
    return mu + torch.randn_like(log_var) * std


class GaussianVAE(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, latent):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor):
        mean, log_var = self.encoder.forward(x)
        device = self.encoder.conv1[0].weight.device
        z_given_x = gaussian_sample(mean, log_var)
        x_hat = self.decoder.forward(z_given_x)
        return mean, log_var, x_hat
