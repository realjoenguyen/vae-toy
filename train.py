import argparse
from collections import deque
import math
import pathlib
import numpy as np
import os
import torch
import tqdm
import wandb

# from PIL import Image

import torch.distributions as dist
import torch.optim as optim

# import torchvision.transforms as transforms

# from categoricalVAE import Encoder, Decoder, CategoricalVAE
import cateVAE
import gaussianVAE

use_wandb = True

if use_wandb:
    wandb_run = wandb.init(
        project="categorical-vae",
        entity="osu-rl",
    )
    training_run_id = wandb_run.id
    # path of the parent of the current file
    parent_path = pathlib.Path(__file__).parent
    wandb.run.log_code(parent_path, exclude_fn=lambda x: "jax" in x or ".history" in x)
else:
    training_run_id = "default"


import torch.utils.data


# def symbolic_to_multihot(layers):
#     n_entities = 17
#     layers = layers.astype(int)
#     new_ob = np.maximum.reduce(
#         [np.eye(n_entities)[layers[..., i]] for i in range(layers.shape[-1])]
#     )
#     new_ob[:, :, 0] = 0
#     return new_ob

import torch


def symbolic_to_multihot(layers):
    # reshape such that images has (num_images, height, width, channels) from (num_images, channels, height, width)
    layers = layers.permute(0, 2, 3, 1)

    n_entities = 17
    # Ensure layers is a torch tensor of type long (integers)
    layers = layers.long()
    # Create a list of one-hot encoded tensors for each layer
    one_hots = [
        torch.eye(n_entities, device=layers.device)[layers[..., i]]
        for i in range(layers.shape[-1])
    ]
    # Use torch.stack to combine the one-hot encodings and then use torch.max to get the multi-hot encoding
    stacked_one_hots = torch.stack(
        one_hots, dim=-2
    )  # Stack along a new dimension to keep individual one-hot encodings separate
    new_ob, _ = torch.max(
        stacked_one_hots, dim=-2
    )  # Use torch.max to combine the encodings across the new dimension
    # Zero out the first channel if needed
    new_ob[:, :, :, 0] = 0
    # reshape back to (num_images, channels, height, width)
    new_ob = new_ob.permute(0, 3, 1, 2)
    print("nonzeros", torch.nonzero(new_ob[0]))
    return new_ob


def load_training_data(dense):
    images = np.load("small_images_s1.npy") if dense else np.load("images_s1.npy")
    images = np.transpose(images, (0, 3, 1, 2))
    mean = np.mean(images)
    std = np.std(images)
    print("normalize mean and std to normal distribution", mean, std)
    print(f"load training data: {images.shape}")
    images = torch.tensor(images, dtype=torch.float)
    if dense:
        multihots = symbolic_to_multihot(images)
        print("shape of multihots", multihots.shape)
        return torch.utils.data.TensorDataset(images, multihots), mean, std
    else:
        return torch.utils.data.TensorDataset(images), mean, std


# import torchvision
# from torchvision import transforms


def load_MNIST():
    transform = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )


def kl_cate(logits: torch.Tensor, device) -> torch.Tensor:
    # logits : B, N, K
    B, N, K = logits.shape
    logits = logits.view(B * N, K)
    EPS = 1e-10
    q = dist.Categorical(logits=logits + EPS)
    p = dist.Categorical(probs=torch.full((B * N, K), 1.0 / K).to(device))
    kl = dist.kl.kl_divergence(q, p)  # b, n
    return kl.view(B, N)


# def create_random_image(
#     model: CategoricalVAE,
#     N: int,
#     K: int,
#     temperature: float,
#     step: int,
#     output_dir: str,
#     device,
# ):
#     random_image = model.generate_random_image(
#         N, K, temperature=temperature, device=device
#     )
#     pil_image = make_pil_image(random_image)
#     pil_image.save(os.path.join(output_dir, f"random_step_{step}.png"))
#     return pil_image


# def make_pil_image(img_tensor: torch.Tensor):
#     img_tensor = img_tensor.detach().cpu().numpy().squeeze()
#     # random_image = (img_tensor * 255).astype(np.uint8)
#     # put through sigmoid - using func from numpy
#     random_image = (1 / (1 + np.exp(-img_tensor))).astype(np.uint8)
#     return Image.fromarray(random_image)


def frange_cycle_linear(n_epoch, start=0, stop=1, n_cycle=30, ratio=0.5):
    L = np.ones(n_epoch) * stop
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule
    print(f"Step={step} for each cycle")

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1

    # assert max L <= top
    assert max(L) <= stop, f"max(L)={max(L)} stop={stop}"
    return L


def force_cudnn_initialization(device):
    s = 32
    dev = torch.device(device)
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
    )


def generate_random_image_cate(N: int, K: int, decoder: cateVAE.Decoder, device):
    NUM = 10
    z = np.zeros((NUM, N, K))
    z = z.reshape(-1, K)
    z[range(NUM * N), np.random.choice(K, NUM * N)] = 1
    z = z.reshape(NUM, N, K)
    with torch.no_grad():
        # (num, 10, 10, 17)
        random_image = decoder.forward(torch.from_numpy(z).float().to(device))

    random_image = torch.sigmoid(random_image) > 0.5
    # take the top 4 pixels for each image, and set the rest to 0
    # flatten = random_image.view(NUM, -1)
    # values, indices = torch.topk(flatten, 4, dim=-1)
    # flatten.fill_(0)
    # # fill back 1
    # flatten.scatter_(1, indices, torch.ones_like(values))
    # random_image = random_image.view(NUM, 17, 10, 10)

    return random_image


def main() -> None:
    # add arguments here using argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", type=str, default="layer")
    parser.add_argument("--dense", type=bool, default=False)
    parser.add_argument("--mean", type=bool, default=False)
    parser.add_argument("--kl_weight", type=str, default="increase")
    # make sure latent is in ['soft', 'hard', 'sample']
    parser.add_argument("--latent", type=str, default="soft")
    parser.add_argument("--kl_min", type=bool, default=True)
    parser.add_argument("--skip", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--load_from", type=str, default="none")
    parser.add_argument("--top_kl_weight", type=float, default=1)
    # parser.add_argument("--latent_type", type=str, default="cate")
    args = parser.parse_args()
    assert args.latent in ["soft", "hard", "gau"]
    # assert args.latent_type in ["gau", "cate"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("device=", device)
    force_cudnn_initialization(device)
    LOG_INTERVAL = 50
    MODEL_SAVE_INTERVAL = 5_000
    BATCH_SIZE = 1_000
    MAX_STEPS = 100_000
    initial_learning_rate = 1e-3
    INIT_TEMP = 1.0
    MIN_TEMP = 0.5
    ANNEAL_RATE = 0.00002
    INCREASE_INTERVAL = 2000
    TEMP_INTERVAL = 2000
    K = 32  # number of classes
    N = 32  # number of categorical distributions
    # set seed
    torch.manual_seed(0)
    np.random.seed(0)

    training_images, MEAN, STD = load_training_data(args.dense)
    train_dataset = torch.utils.data.DataLoader(
        dataset=training_images,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    image_shape = next(iter(train_dataset))[0][0].shape  # [1, 28, 28]
    print(f"image_shape={image_shape}")
    output_shape = torch.Size([17, 10, 10])
    if args.latent == "gau":
        encoder = gaussianVAE.Encoder(
            N, image_shape, skip=args.skip, norm=args.norm, dense=args.dense
        )
        decoder = gaussianVAE.Decoder(N, output_shape, skip=args.skip, norm=args.norm)
        model = gaussianVAE.GaussianVAE(encoder, decoder, args.latent).to(device)
    else:
        encoder = cateVAE.Encoder(
            N, K, image_shape, norm=args.norm, dense=args.dense, skip=args.skip
        )
        decoder = cateVAE.Decoder(N, K, output_shape, norm=args.norm, skip=args.skip)
        model = cateVAE.CategoricalVAE(encoder, decoder, args.latent).to(device)

    if args.load_from != "none":
        model.load_state_dict(torch.load(args.load_from))

    if use_wandb:
        wandb.watch(model, log_freq=LOG_INTERVAL)
        # log args to wandb
        wandb.config.update(args)
        wandb.config.update(
            {
                "LOG_INTERVAL": LOG_INTERVAL,
                "MODEL_SAVE_INTERVAL": MODEL_SAVE_INTERVAL,
                "BATCH_SIZE": BATCH_SIZE,
                "MAX_STEPS": MAX_STEPS,
                "initial_learning_rate": initial_learning_rate,
                "INIT_TEMP": INIT_TEMP,
                "MIN_TEMP": MIN_TEMP,
                "ANNEAL_RATE": ANNEAL_RATE,
                "INCREASE_INTERVAL": INCREASE_INTERVAL,
                "TEMP_INTERVAL": TEMP_INTERVAL,
                "K": K,
                "N": N,
            }
        )

    parameters = list(model.parameters())
    optimizer = optim.Adam(parameters, lr=initial_learning_rate)
    # learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    step = 0
    temp = INIT_TEMP

    # make folder for images
    output_dir = os.path.join("outputs", training_run_id)
    os.makedirs(output_dir, exist_ok=True)

    progress_bar = tqdm.tqdm(total=MAX_STEPS, desc="Training")
    kl_weights = frange_cycle_linear(70000, stop=args.top_kl_weight)
    # kl_weights =
    table_data = deque(maxlen=30)
    while step < MAX_STEPS:
        for data in train_dataset:
            # x should be a batch of torch.Tensor inputs, of shape [B, F, T]
            if args.dense:
                x, y = data
                y = y.to(device)
                x = x.to(device)
                labels = y
            else:
                x = data[0]
                x = x.to(device)
                labels = x
            x_normalized = (x - MEAN) / STD
            if args.latent == "gau":
                mean, log_var, x_hat = model.forward(x_normalized)
                kl_loss = torch.mean(
                    torch.sum(
                        0.5 * (-log_var + torch.exp(log_var) + mean**2 - 1), dim=-1
                    )
                )
            # else args.latent_type == "cate":
            else:
                logits, x_hat = model.forward(x_normalized, temp)
                kl_loss = torch.mean(torch.sum(kl_cate(logits, device), dim=-1))

            # use focal loss instead
            # recon_loss = (
            #     sigmoid_focal_loss(x_hat, x, alpha=0.25, gamma=2).sum() / x.shape[0]
            # )
            recon_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    x_hat, labels, reduction="none"
                ).sum()
            ) / x.shape[0]

            if args.kl_weight == "increase":
                kl_weight = min(1, step / INCREASE_INTERVAL)
            elif args.kl_weight == "cosine":
                kl_weight = kl_weights[step]
            else:
                kl_weight = float(args.kl_weight)
            assert (
                kl_weight <= args.top_kl_weight
            ), f"kl_weight={kl_weight} top_kl_weight={args.top_kl_weight}"

            if args.kl_min:
                loss = kl_weight * max(1, kl_loss) + recon_loss
            else:
                loss = kl_weight * kl_loss + recon_loss

            progress_bar.set_description(
                f"Recon loss = {recon_loss:.2f} / KL loss = {kl_loss:.2f}, KL weight {kl_weight:.2f}, Temp {temp:.2f}, Elbo {recon_loss + kl_loss:.2f}"
            )
            gradnorm = torch.nn.utils.clip_grad_norm_(parameters, 10)
            # find var of gradnorm
            loss.backward()
            optimizer.step()

            # Incrementally anneal temperature and learning rate.
            if step % TEMP_INTERVAL == 1:
                temp = np.maximum(
                    INIT_TEMP * np.exp(-ANNEAL_RATE * step),
                    MIN_TEMP,
                )

            if step % LOG_INTERVAL == 0:
                if use_wandb:
                    if args.latent != "gau":
                        image_random = generate_random_image_cate(N, K, decoder, device)
                        image_random_str = image2str(image_random)
                        image_gt_str = image2str(x[:10])
                        # image_recon = torch.bernoulli(torch.sigmoid(x_hat[:10]))
                        image_recon = (torch.sigmoid(x_hat[:10]) > 0.5).float()
                        image_recon_str = image2str(image_recon)

                    wandb.log(
                        {
                            "grad_norm": gradnorm,
                            "kl_loss": kl_loss,
                            "reconstruction_loss": recon_loss,
                            "loss": loss,
                            "elbo": recon_loss + kl_loss,
                            "kl_weight": kl_weight,
                            # "random_image": wandb.Image(random_image),
                            # "x": wandb.Image(make_pil_image(x[0])),
                            # "x_hat": wandb.Image(make_pil_image(x_hat[0])),
                            "temperature": temp,
                            # "phi_hist": wandb.Histogram(
                            #     phi.exp().detach().numpy().flatten()
                            # ),
                            # "phi_sum_hist": wandb.Histogram(
                            #     phi.exp().sum(axis=2).detach().numpy().flatten()
                            # ),
                            "x_hat_hist": wandb.Histogram(
                                x_hat.detach().cpu().numpy().flatten()
                            ),
                            "x_hist": wandb.Histogram(
                                x.detach().cpu().numpy().flatten()
                            ),
                            # "learning_rate": learning_rate_scheduler.get_lr(),
                        },
                        step=step,
                    )

                    if args.latent != "gau":
                        table_data.append(
                            [
                                loss.item(),
                                kl_loss,
                                recon_loss,
                                image_gt_str,
                                image_recon_str,
                                image_random_str,
                            ]
                        )
                        wandb.log(
                            {
                                "table": wandb.Table(
                                    columns=[
                                        "loss",
                                        "kl",
                                        "recon",
                                        "image_gt",
                                        "image_recon",
                                        "image_imagine",
                                    ],
                                    data=list(table_data),
                                )
                            },
                            step=step,
                        )

            if (step + 1) % MODEL_SAVE_INTERVAL == 0:
                torch.save(
                    model.state_dict(), os.path.join(output_dir, f"save_{step}.pt")
                )

            step += 1
            progress_bar.update(1)


def image2str(image: torch.Tensor):
    res = ""
    N_PIXELS = 10
    # covert to numpy
    image = image.cpu().detach().numpy()
    for t in range(min(10, image.shape[0])):
        # print all nonzero pixels
        # format: (array([i, i, ...]), array([j, j, ...]), array([k, k, ...]))
        nonzeros = np.nonzero(image[t])
        nonzeros_count = len(nonzeros[0])
        res += f"Image {t}, nonzero={nonzeros_count}"
        res += "\n"
        for i, j, k in list(zip(*nonzeros))[:N_PIXELS]:
            res += f"Pixel {i}, {j}, {k}, {image[t][i, j, k]}\n"
        res += "\n"
    return res


if __name__ == "__main__":
    main()
