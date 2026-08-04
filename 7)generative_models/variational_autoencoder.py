#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchvision

import npfl138
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--decoder_layers", default=[500, 300], type=int, nargs="+", help="Decoder layers.")
parser.add_argument("--encoder_layers", default=[300, 500], type=int, nargs="+", help="Encoder layers.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        return image, image


class VAE(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._seed = args.seed
        self._z_dim = args.z_dim

        self._z_prior = lambda: torch.distributions.Normal(  # Lambda method to construct the
            torch.zeros(args.z_dim, device=self.device),     # prior distribution on the current
            torch.ones(args.z_dim, device=self.device))      # device of the model.

        # Construct the encoder with more layers
        self.encoder = torch.nn.Sequential(torch.nn.Flatten())
        input_units = MNIST.C * MNIST.H * MNIST.W
        for i in range(len(args.decoder_layers)):
            self.encoder.append( torch.nn.Linear(in_features= input_units, out_features=args.decoder_layers[i]))
            self.encoder.append( torch.nn.ReLU())
            input_units = args.decoder_layers[i]
        self.encoder.append(torch.nn.Linear(in_features= input_units, out_features= 2 * args.z_dim ))

        # Construct the decoder
        self.decoder = torch.nn.Sequential()
        input_units = args.z_dim
        for i in range(len(args.decoder_layers)):
            self.decoder.append( torch.nn.Linear(in_features= input_units, out_features=args.decoder_layers[i]))
            self.decoder.append( torch.nn.ReLU())
            input_units = args.decoder_layers[i]

        self.decoder.append(torch.nn.Linear(in_features= input_units, out_features= MNIST.C * MNIST.H * MNIST.W))
        self.decoder.append(torch.nn.Sigmoid())
        self.decoder.append(torch.nn.Unflatten(dim= 1, unflattened_size=(MNIST.C, MNIST.H, MNIST.W)))

    def train_step(self, xs: tuple[torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        # Pass images throught encoder
        images = xs[0]
        encoder_logits = self.encoder(images)

        # compute mean and std from the logits
        z_mean = encoder_logits[ :, : self._z_dim]
        z_std = torch.exp(encoder_logits[:, self._z_dim : ])

        distribution = torch.distributions.Normal(loc=z_mean, scale=z_std)
        # sample from the encoded distribution
        z = distribution.rsample()
        decoded = self.decoder(z)

        # compute loss as a combination of reconstruction and latent loss
        reconstruction_loss = torch.nn.functional.binary_cross_entropy(decoded, images)
        latent_loss = torch.distributions.kl.kl_divergence(p = distribution, q = self._z_prior()).mean()
        loss = reconstruction_loss * MNIST.H * MNIST.W * MNIST.C + latent_loss * self._z_dim

        self.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()

        # Return the mean of the overall loss, and the current reconstruction and latent losses.
        # loss = self.loss_tracker(loss)
        return {"loss": loss, "reconstruction_loss": reconstruction_loss, "latent_loss": latent_loss}

    def generate(self, epoch: int, logs: dict[str, float]) -> None:
        GRID = 20
        with torch.no_grad(), torch.device(self.device):
            # Generate GRIDxGRID images.
            random_images = self.decoder(self._z_prior().sample([GRID * GRID]))

            # Generate GRIDxGRID interpolated images.
            if self._z_dim == 2:
                # Use 2D grid of Z values for interpolated images.
                starts = torch.stack([-2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
                ends = torch.stack([2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
            else:
                # Otherwise generate random Z for the first and the last column.
                starts, ends = self._z_prior().sample([2, GRID])
            interpolated_z = torch.cat(
                [starts[i] + (ends[i] - starts[i]) * torch.linspace(0., 1., GRID).unsqueeze(-1) for i in range(GRID)])
            interpolated_images = self.decoder(interpolated_z)

            # Stack the random images, then an empty row, and finally interpolated images.
            grid = torchvision.utils.make_grid(
                list(random_images) + list(torch.zeros([GRID, MNIST.C, MNIST.H, MNIST.W])) + list(interpolated_images),
                nrow=GRID, padding=0)
            self.get_tb_writer("train").add_image("images", grid, epoch)


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data and create dataloaders.
    mnist = MNIST(args.dataset, sizes={"train": args.train_size})
    train = TrainableDataset(mnist.train).dataloader(args.batch_size, shuffle=True, seed=args.seed)

    # Create the model and train it.
    model = VAE(args)
    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        logdir=args.logdir,
    )

    logs = model.fit(train, epochs=args.epochs, callbacks=[VAE.generate])


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
