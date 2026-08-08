#!/usr/bin/env python3
import argparse
import copy
import datetime
import os
import re

import torch

import npfl138
from npfl138.datasets.image64_dataset import Image64Dataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--dataset", default="oxford_flowers102", type=str, help="Image64 dataset to use.")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")
parser.add_argument("--epoch_batches", default=1_000, type=int, help="Batches per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--loss", default="L1Loss", type=str, help="The loss to use.")
parser.add_argument("--plot_each", default=None, type=int, help="Plot generated images every such epoch.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--sigma_min", default=0.001, type=float, help="Sigma_min used in OT paths.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stages", default=4, type=int, help="Stages to use.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# The diffusion model architecture building blocks.
class SinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal embeddings used to embed the current time step."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[-1] == 1

        # Create the frequency terms
        half_dim = self.dim // 2
        device = inputs.device
        dtype = inputs.dtype
        i = torch.arange(half_dim, device=device, dtype=dtype)

        # Compute denominator: 20 ** (2 * i / dim)
        denom = 20 ** (2 * i / self.dim)
        scaled_inputs = 2 * torch.pi * inputs / denom

        # Compute sinusoidal embeddings
        sin = torch.sin(scaled_inputs)
        cos = torch.cos(scaled_inputs)
        embedding = torch.cat([sin, cos], dim=-1)
        return embedding


class ResidualBlock(torch.nn.Module):
    """A residual block with two 3x3 convolutions and a time embedding."""
    def __init__(self, width: int) -> None:
        super().__init__()
        self.cnn1 = torch.nn.Sequential(
            torch.nn.LazyConv2d(width, 3, padding="same", bias=False),
            torch.nn.GroupNorm(min(width // 4, 16), width),
            torch.nn.SiLU()
        )

        self.time_embedding = torch.nn.Sequential(
            torch.nn.LazyLinear(width),
            torch.nn.SiLU(),
        )

        num_groups = min(width // 4, 16)
        self.group_norm = torch.nn.GroupNorm(num_groups, width)
        torch.nn.init.zeros_(self.group_norm.weight)

        self.cnn2 = torch.nn.Sequential(
            torch.nn.LazyConv2d(width, 3, padding="same", bias=False),
            self.group_norm
        )

    def forward(self, images: torch.Tensor, times: torch.Tensor) -> torch.Tensor:

        logits = self.cnn1(images)
        time_logits = self.time_embedding(times)

        logits += time_logits[:, :, None, None]
        logits = self.cnn2(logits)

        return images + logits


class DownscalingBlock(torch.nn.Module):
    """Downscaling block returning both the features of original and downscaled size."""
    def __init__(self, residual_blocks: int, width: int) -> None:
        super().__init__()
        self.residual = torch.nn.Sequential(
            *[ResidualBlock(width) for i in range(residual_blocks)]
        )
        self.output = torch.nn.Sequential(
            torch.nn.LazyConv2d(width << 1, 3, stride=2, padding=1),
        )


    def forward(self, images: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = images
        for block in self.residual:
            x = block(x, times)
        residual = x
        downscaled = self.output(residual)
        return downscaled, residual


class UpscalingBlock(torch.nn.Module):
    """Upscaling block using a skip connection from the corresponding downscaling block."""
    def __init__(self, residual_blocks: int, width: int) -> None:
        super().__init__()
        self.upscale = torch.nn.Sequential(
            torch.nn.LazyConvTranspose2d(width, 4, stride=2, padding=1),
        )

        self.residual_processing = torch.nn.Sequential(
            torch.nn.LazyConv2d(width, 3, padding="same"),
        )

        self.res_blocks = torch.nn.Sequential(
            *[ResidualBlock(width) for i in range(residual_blocks)]
        )

    def forward(self, images: torch.Tensor, skip_connections: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        upscaled = self.upscale(images)
        skip = self.residual_processing(skip_connections)
        upscaled += skip

        x = upscaled
        for block in self.res_blocks:
            x = block(x, times)
        return x

class UNet(torch.nn.Module):
    """The U-Net architecture used in the flow matching model."""
    def __init__(self, channels: int, stage_blocks: int, stages: int) -> None:
        super().__init__()
        self.time_embedding = SinusoidalEmbedding(channels)
        self.cnn1 = torch.nn.Sequential(
            torch.nn.LazyConv2d(channels, 3, padding="same"),
        )
        self.downscaling_blocks = torch.nn.ModuleList(
            [DownscalingBlock(stage_blocks, channels << i) for i in range(stages)]
        )
        self.middle = torch.nn.Sequential(
            *[ResidualBlock(channels << stages) for i in range(stage_blocks)]
        )
        self.upscaling_blocks = torch.nn.ModuleList(
            [UpscalingBlock(stage_blocks, channels << (stages -1 - i)) for i in range(stages)]
        )
        self.output = torch.nn.Sequential(
            torch.nn.LazyConv2d(Image64Dataset.C, 3, padding="same"),
        )


    def forward(self, images: torch.Tensor, times: torch.Tensor) -> None:
        times_embedded = self.time_embedding(times)
        logits = self.cnn1(images)

        ds_outputs = []
        for block in self.downscaling_blocks:
            logits, ds_output = block(logits, times_embedded)
            ds_outputs.append(ds_output)

        for block in self.middle:
            logits = block(logits, times_embedded)

        for block, ds_output in zip(self.upscaling_blocks, reversed(ds_outputs)):
            logits = block(logits, ds_output, times_embedded)

        logits = self.output(logits)
        return logits


class FlowMatching(npfl138.TrainableModule):
    """The model used for flow matching, capable of generating images."""
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._model = UNet(args.channels, args.stage_blocks, args.stages)

        self._ema_model = None
        self._ema_momentum = args.ema
        self._sigma_min = args.sigma_min
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]))

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Method to normalize the input image to have a standard distribution."""
        image = (image - self.imagenet_mean[None, :, None, None]) / self.imagenet_std[None, :, None, None]
        return image

    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """The inverse of the `normalize_image` method."""
        image = image * self.imagenet_std[None, :, None, None] + self.imagenet_mean[None, :, None, None]
        return image

    def train_step(self, xs: tuple[torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Perform a single training update."""
        # Unpack the input batch.
        images = xs[0]

        # Generate random noise and random time steps.
        noises = torch.randn_like(images)
        times = torch.rand(images.shape[0], 1, device=images.device)

        # Normalize the images.
        images_norm = self.normalize_image(images)

        # Compute interpolated images according to optimal transport interpolation
        times_bc = times[..., None, None]
        noisy_images = (1 - (1 - self._sigma_min) * times_bc )* noises + times_bc* images_norm

        # Predict the vector field from the interpolated images and time steps.
        predicted = self._model(noisy_images, times)

        # Compute the target vector field according to d(noisy_images)/dt
        target_field = images_norm - (1 - self._sigma_min) * noises

        loss = self.loss(predicted, target_field)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            self.optimizer.step()
            if self._ema_model is None:
                self._ema_model = copy.deepcopy(self._model)
                self._ema_model.requires_grad_(False)

            for ema_variable, variable in zip(self._ema_model.parameters(), self._model.parameters()):
                ema_variable.mul_(self._ema_momentum).add_(variable * (1 - self._ema_momentum) )

            return {"loss": self.track_loss(loss)}

    @torch.no_grad()
    def generate(self, initial_noise: torch.Tensor, steps: int) -> torch.Tensor:
        images = initial_noise.to(self.device)
        trajectory = []

        # apply euler method and find xt's for every time step
        for i in range(steps):
            t = torch.full((images.shape[0], 1), i / steps, device=images.device)
            flow = self._ema_model(images, t)
            images = images + (1 / steps) * flow
            trajectory.append(images.clone())

        # Apply the denormalization to the generated images and the trajectory.
        return self.denormalize_image(images), list(map(self.denormalize_image, trajectory))


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        return image, image  # return the image both as the input and the target


class FixedNumberOfSamples(torch.utils.data.Sampler):
    def __init__(self, size: int, samples: int, seed: int) -> None:
        self._size, self._samples, self._permutation = size, samples, []
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self._samples

    def __iter__(self):
        for _ in range(self._samples):
            self._permutation = self._permutation or torch.randperm(self._size, generator=self._generator).tolist()
            yield self._permutation[0]
            self._permutation = self._permutation[1:]


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ))

    # Load the image data.
    images64 = Image64Dataset(args.dataset)
    train = TrainableDataset(images64.train).dataloader(args.batch_size, sampler=FixedNumberOfSamples(
        len(images64.train), args.batch_size * args.epoch_batches, args.seed))

    # Create the model.
    flow_matching = FlowMatching(args)

    # Class for sampling images and storing them in TensorBoard.
    class TBSampler:
        def __init__(self, columns: int, rows: int, seed: int) -> None:
            self._columns = columns
            self._rows = rows
            self._noise = torch.randn(
                rows, columns, Image64Dataset.C, Image64Dataset.H, Image64Dataset.W,
                generator=torch.Generator().manual_seed(seed)
            )

        @torch.no_grad()
        def __call__(self, model, epoch, logs) -> None:
            # After the last epoch and every `args.plot_each` epoch, generate a sample to TensorBoard logs.
            if epoch == args.epochs or epoch % (args.plot_each or args.epochs) == 0:
                # Generate a grid of `self._columns *  self._rows` independent samples.
                rows = [model.generate(noise, args.sampling_steps)[0] for noise in list(self._noise)]
                images = torch.cat([torch.cat(list(row), dim=-1) for row in rows], dim=-2)
                model.get_tb_writer("train").add_image("images", images, epoch)
                # Generate gradual denoising process for `rows` samples, showing `self._columns` steps.
                steps = max(1, args.sampling_steps // (self._columns - 1))
                samples, process = model.generate(self._noise[:, 0], steps * (self._columns - 1))
                process = torch.cat([torch.cat(list(col), dim=-2) for col in process[::steps] + [samples]], dim=-1)
                model.get_tb_writer("train").add_image("process", process, epoch)
            # After the last epoch, store statistics of the generated sample for ReCodEx to evaluate.
            if epoch == args.epochs:
                images = images.numpy(force=True)
                logs["sample_mean"], logs["sample_std"] = images.mean(), images.std()

    # Train the model.
    flow_matching.configure(
        optimizer=torch.optim.AdamW(flow_matching.parameters()),
        loss=getattr(torch.nn, args.loss)(),
        logdir=args.logdir
    )
    logs = flow_matching.fit(train, epochs=args.epochs, callbacks=[TBSampler(16, 10, args.seed)])
    return logs


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
