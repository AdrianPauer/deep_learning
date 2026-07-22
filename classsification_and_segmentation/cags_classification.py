#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import timm
import torch
import torchvision.transforms.v2 as v2
import torchmetrics
import matplotlib.pyplot as plt

import npfl138
from npfl138.datasets.cags import CAGS

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CAGS.Dataset, preprocessing=None) -> None:
        super().__init__(dataset)
        self._preprocessing = preprocessing

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask, label = example['image'], example['mask'], example['label']
        if self._preprocessing is not None: 
            img = self._preprocessing(img)
        return img, label

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, efficient_model) -> None:
        super().__init__()
        self._args = args

        self.eff_model = efficient_model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1280,CAGS.LABELS)
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            eff_output = self.eff_model(inputs)
        result = self.model(eff_output)

        return result
    
    def params(self):
        return self.model.parameters()
    
    def train(self, mode=True):
        self.model.train(mode) 
        self.eff_model.eval()  
        return self

    def eval(self):
        self.model.eval()  
        self.eff_model.eval()  
        return self


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # load the dataset
    cags = CAGS(decode_on_demand=False)
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    train = torch.utils.data.DataLoader(TransformedDataset(cags.train, preprocessing), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(TransformedDataset(cags.dev, preprocessing), batch_size=args.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(TransformedDataset(cags.test, preprocessing), batch_size=args.batch_size, shuffle=False)

    model = Model(args, efficientnetv2_b0)
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=cags.LABELS)},
        logdir=args.logdir,
    )

    model.fit(train, dev=dev, epochs=args.epochs)

    # visualize predictions on dev set
    counter = 0
    for  batch, gold_labels in dev:
        with torch.no_grad():
            predictions = model(batch)

        predictions = torch.topk(predictions.softmax(dim=-1), k=3)
        for i in range(batch.size(0)):
            # undo normalization
            mean = torch.tensor(efficientnetv2_b0.pretrained_cfg["mean"]).view(3, 1, 1)
            std = torch.tensor(efficientnetv2_b0.pretrained_cfg["std"]).view(3, 1, 1)

            img = batch[i].cpu() * std + mean
            img = img.clamp(0, 1)

            img = img.permute(1, 2, 0)

            probs = predictions[0][i]
            labels = predictions[1][i]
            gold_label = gold_labels[i]

            plt.figure(figsize=(8, 5))
            plt.imshow(img)
            plt.axis("off")

            title = "\n".join(
                f"{CAGS.LABEL_NAMES[label]} ({prob:.2})"
                for prob, label in zip(probs,labels)
            )
            title = f"gold : {CAGS.LABEL_NAMES[gold_label]} \n" + title

            if gold_label == labels[0]: color = "green"
            else:
                color = "red"
                print(counter)

            plt.title(title, fontsize=9, color=color)
            plt.savefig(f"pred_images/prediction_{counter}.png", bbox_inches="tight", dpi=200)
            plt.close()
            counter += 1

    # annotations on test set
    # os.makedirs(args.logdir, exist_ok=True)
    # with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
    #     for prediction in model.predict(test, data_with_labels=True):
    #         print(np.argmax(prediction), file=predictions_file)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
