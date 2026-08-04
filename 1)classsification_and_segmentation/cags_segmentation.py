#!/usr/bin/env python3
import argparse
import os

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2

import npfl138
from npfl138.datasets.cags import CAGS

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--logdir", default="logs_segmentation", type=str)

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CAGS.Dataset, preprocessing=None) -> None:
        super().__init__(dataset)
        self._preprocessing = preprocessing

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask, label = example['image'], example['mask'], example['label']
        if self._preprocessing is not None: 
            img = self._preprocessing(img)
        
        return img, mask

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, efficient_model) -> None:
        super().__init__()
        self._args = args

        # backbone
        self.eff_model = efficient_model

        # upscalings blocks
        self.up1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1280, 112, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(112),
            torch.nn.ReLU()
            )

        self.right1up = torch.nn.Sequential(
            torch.nn.Conv2d(224, 48, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 48, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(48,  48, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU()
        )
        self.right2up = torch.nn.Sequential(
            torch.nn.Conv2d(96, 32, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32,  32, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )

        self.right3up = torch.nn.Sequential(
            torch.nn.Conv2d(64, 16, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16,  16, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.right4up = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16,  16, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16,  1, kernel_size=1, padding='same'),
            torch.nn.Sigmoid()
        )


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output, features = self.eff_model.forward_intermediates(inputs)
        # features is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
        up1 = self.up1(output)
        concatenated1 = torch.cat([features[3], up1], dim=1)

        right1_up2 = self.right1up(concatenated1)
        concatenated2 = torch.cat([features[2], right1_up2], dim=1)

        right2_up3 = self.right2up(concatenated2)
        concatenated3 = torch.cat([features[1], right2_up3], dim=1)

        right3_up4 = self.right3up(concatenated3)
        concatenated4 = torch.cat([features[0], right3_up4], dim=1)

        result = self.right4up(concatenated4)
        return result
    
    
    def train(self, mode=True):
        self.eff_model.eval()  
        return self

    def eval(self):
        self.eff_model.eval()  
        return self


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    cags = CAGS(decode_on_demand=False)
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # build data loaders
    train = torch.utils.data.DataLoader(TransformedDataset(cags.train, preprocessing), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(TransformedDataset(cags.dev, preprocessing), batch_size=args.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(TransformedDataset(cags.test, preprocessing), batch_size=args.batch_size, shuffle=False)

    # define a model
    model = Model(args, efficientnetv2_b0)
    
    # freeze backbone model
    for param in model.eff_model.parameters():
            param.requires_grad = False 

    model.configure(
        optimizer=torch.optim.AdamW(model.parameters()),
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics={"accuracy": CAGS.MaskIoUMetric()},
        logdir=args.logdir,
    )

    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        for mask in model.predict(test, data_with_labels=True):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
