from typing import Tuple

import torch

# import torchvision

import torch.distributions as dist


def gumbel_distribution_sample(shape: torch.Size, eps=1e-20, device="cuda"):
    # shape: (b*n, k)
    U = torch.rand(shape).to(device)
    # from uniform [0, 1] to gumbel
    return -torch.log(-torch.log(U + eps) + eps)


from torch import nn
import torch.nn.functional as F


def gumbel_softmax(
    logits: torch.Tensor, temperature: float, batch=False, hard=False, device="cuda"
):
    """
    Gumbel-softmax.
    input: [*, n_classes] (or [b, *, n_classes] for batch)
    return: flatten --> [*, n_class] a one-hot vector (or b, *, n_classes for batch)
    """
    input_shape = logits.shape
    if batch:
        assert len(logits.shape) == 3
        b, n, k = input_shape
        logits = logits.view(b * n, k)
    assert len(logits.shape) == 2  # b*n, k
    y = logits + gumbel_distribution_sample(logits.shape, device=device)
    y = F.softmax(y / temperature, dim=-1)

    if hard:
        # straight through
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=-1).unsqueeze(-1), 1)
        y = (y_hard - y).detach() + y

    return y.view(input_shape)


from torch import nn


class Encoder(torch.nn.Module):
    cnn: torch.nn.Module
    input_shape: torch.Size
    N: int  # number of categorical distributions
    K: int  # number of classes

    def __init__(
        self,
        N: int,
        K: int,
        input_shape: torch.Size,
        skip=False,
        norm="layer",
        dense=False,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.input_shape = input_shape
        print("N =", N, "and K =", K)
        # turn it to sequential
        input_channel = 2 if dense else 17
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64) if norm == "batch" else nn.LayerNorm([64, 5, 5]),
            nn.SiLU(),
        )

        # turn it to sequential
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128) if norm == "batch" else nn.LayerNorm([128, 3, 3]),
            nn.SiLU(),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 128, N * K),
            nn.BatchNorm2d(N * K) if norm == "batch" else nn.LayerNorm([N * K]),
            nn.SiLU(),
        )

        # use skip connection
        self.skip = skip
        if self.skip:
            self.resample = nn.Sequential(
                # turn 10 x 10 x 17 to 3 x 3 x 128
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
        x = self.fc(x)
        x = x.reshape(-1, self.N, self.K)
        return x


import torch.nn.functional as F


class Decoder(torch.nn.Module):
    output_shape: torch.Size
    N: int  # number of categorical distributions
    K: int  # number of classes

    def __init__(
        self,
        N: int,
        K: int,
        output_shape: torch.Size,
        skip=True,
        norm="batch",
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.output_shape = output_shape

        # First main block (Transpose Convolutions)
        self.fc1 = nn.Linear(N * K, 128 * 3 * 3)
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
        # set bias of conv2 to -3
        # self.deconv2[0].bias.data.fill_(-3)

        self.skip = skip
        if self.skip:
            self.resample = nn.Sequential(
                nn.ConvTranspose2d(128, 17, 1, stride=4, output_padding=1),
                nn.BatchNorm2d(17) if norm == "batch" else nn.LayerNorm([17, 10, 10]),
            )

        # Second main block (Final Linear Layer)
        self.fc2 = nn.Linear(17 * 10 * 10, 17 * 10 * 10)
        # turn bias of this to -1
        self.fc2.bias.data.fill_(-3)

    def forward(self, z):
        x = self.fc1(z.reshape(-1, self.N * self.K)).reshape(-1, 128, 3, 3)
        if self.skip:
            resample = self.resample(x)
        # # Preparing for the linear layer
        # x = x.view(x.size(0), -1)  # Flatten

        # # Second main block processing
        x = self.deconv1(x)
        x = self.deconv2(x)

        if self.skip:
            x = x + resample

        x = self.fc2((x).view(-1, 17 * 10 * 10))
        x_hat = x.view((-1,) + self.output_shape)

        return x_hat


class CategoricalVAE(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, latent):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = 1.0
        self.latent = latent

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        logits = self.encoder.forward(x)
        B, N, K = logits.shape
        device = self.encoder.conv1[0].weight.device
        z_given_x = gumbel_softmax(logits, temperature, batch=True, hard=self.latent == "hard", device=device)  # type: ignore
        x_hat = self.decoder.forward(z_given_x)
        return logits, x_hat

        # np_y = np.zeros((M, categorical_dim), dtype=np.float32)
        # np_y[range(M), np.random.choice(categorical_dim, M)] = 1
        # np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
        # sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
        # if args.cuda:
        #     sample = sample.cuda()
        # sample = model.decode(sample).cpu()
        # save_image(sample.data.view(M // latent_dim, 1, 28, 28),
        #            'data/sample_' + str(epoch) + '.png')


# %%
# import torch
import numpy as np


# x_hat = generate_random_image(32, 32, Decoder())
