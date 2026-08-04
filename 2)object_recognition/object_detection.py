#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2
import torchvision

import bboxes_utils
import npfl138
import heads
from npfl138.datasets.svhn import SVHN

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=6, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: SVHN.Dataset, preprocessing=None, anchors = None, predict_mode = False) -> None:
        super().__init__(dataset)
        self._preprocessing = preprocessing
        self.anchors = anchors
        self.predict_mode = predict_mode

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        img, classes, bboxes = example['image'], example['classes'], example['bboxes']
        target_width, target_height = img.shape[1], img.shape[2]
        if self._preprocessing is not None: 
            img = self._preprocessing(img)

        # scale the bboxes to the h,w = 224
        bboxes = bboxes.float()
        scale_x = 224 / target_width
        scale_y = 224 / target_height
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
        orig_params = torch.tensor([target_width, target_height])

        bboxes = bboxes.to(torch.device('cuda'))
        classes = classes.to(torch.device('cuda'))
        img = img.to(torch.device('cuda'))

        if self.predict_mode:
            return orig_params,img
        else:
            anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(
                anchors=self.anchors,
                gold_classes=classes,
                gold_bboxes=bboxes,
                iou_threshold=0.4,
            )
            # uncomment for data visualization (but you have to move each tensor to cpu)
            # fig, ax = plt.subplots(figsize=(8, 6))
            # ax.imshow(img.permute(1, 2, 0).cpu())
            # for box in bboxes:
            #     top, left, bottom, right = box
            #     rect = patches.Rectangle(
            #         [left, top], right - left, bottom - top,
            #         linewidth=2,
            #         edgecolor='red',
            #         facecolor='none'
            #     )
            #     ax.add_patch(rect)
            # plt.show()
            return img, (anchor_bboxes, anchor_classes)
    
    
    
class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, efficient_model) -> None:
        super().__init__()
        self._args = args
        self.eff_model = efficient_model

        for param in self.eff_model.parameters():
            param.requires_grad = False

        # define 2 heads for regression (region of interests) and classification
        self.classification = heads.ClassificationModel(192)
        self.regression = heads.RegressionModel(192)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output, features = self.eff_model.forward_intermediates(inputs)

        # take 5th features from backbone
        C5 = features[4]
        cl_logits = self.classification(C5)
        bb_logits = self.regression(C5)

        return cl_logits, bb_logits
    
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

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # load svhn dataset and backbone model
    svhn = SVHN( decode_on_demand=False)
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
        v2.Resize((224,224))
    ])

    def generate_anchors( grid_size,
                          stride, base_size=32,
                          scales=[1, 2, 4],
                          aspect_ratios=[0.5, 1.0, 2.0]):

        # generate set of anchors with different scale and ratios
        base_anchors = []
        for scale in scales:
            for ratio in aspect_ratios:
                w = base_size * scale * (ratio ** 0.5)
                h = base_size * scale / (ratio ** 0.5)
                
                x_min = -w / 2 + 3.5
                y_min = -h / 2 + 3.5
                x_max = w / 2 + 3.5
                y_max = h / 2 + 3.5
                
                base_anchors.append([x_min, y_min, x_max, y_max])
        
        base_anchors = torch.asarray(base_anchors)
       
        shifts_x = torch.arange(0, grid_size * stride, step=stride)
        shifts_y = torch.arange(0, grid_size * stride, step=stride)
        
        shifts_x, shifts_y = torch.meshgrid(shifts_x, shifts_y, indexing="xy")
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=-1)
        shifts = shifts.reshape(-1, 4)

        all_anchors = base_anchors.unsqueeze(0) + shifts.unsqueeze(1)  # Broadcasting
        all_anchors = all_anchors.view(-1, 4)

        return all_anchors
    
    anchors = generate_anchors(7,32)
    anchors = anchors.to(torch.device('cuda'))

    train = torch.utils.data.DataLoader(TransformedDataset(svhn.train, preprocessing, anchors), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(TransformedDataset(svhn.dev, preprocessing, anchors), batch_size=args.batch_size, shuffle=False)
    dev_testing = torch.utils.data.DataLoader(TransformedDataset(svhn.dev, preprocessing, anchors, predict_mode=True), batch_size=args.batch_size, shuffle=False)
    test = torch.utils.data.DataLoader(TransformedDataset(svhn.test, preprocessing, anchors, predict_mode=True), batch_size=args.batch_size, shuffle=False)

    model = Model(args, efficientnetv2_b0)
    model = model.to(torch.device('cuda'))

    def custom_loss(y_pred, y):
        # logits from heads
        cl_logits, reg_logits = y_pred  # shape cl: [batch, anchors, 11] reg: [batch, anchors, 4]

        # assigned anchor classe according to iou metric, bboxes in rcnn format
        anchor_bboxes, anchor_classes = y
        # foreground anchors
        valid_mask = anchor_classes > 0
        # convert anchor classes to one hot
        target_classes_one_hot = torch.nn.functional.one_hot(anchor_classes, num_classes=11).float()
        # focal loss for classification only for foreground predictions
        classification_loss = torchvision.ops.sigmoid_focal_loss(cl_logits,
                                                                 target_classes_one_hot[:, :, 1:],
                                                                 reduction='sum')
        # smooth l1 loss
        regression_loss = torch.nn.functional.smooth_l1_loss(reg_logits[valid_mask],
                                                             anchor_bboxes[valid_mask],
                                                             reduction="sum")

        total_loss = (classification_loss + regression_loss) / valid_mask.sum()

        return total_loss
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters()),
        loss= custom_loss,
        logdir=args.logdir
    )

    def generate_predictions(model, test_loader, anchors, write_to_file=False):
        # function generates predictions and loss on dev set during the training
        model.eval()
        all_predictions = []
        for true_shapes,imgs in test_loader:
            with torch.no_grad():
                cl_logits, reg_logits = model(imgs)  

                for i in range(reg_logits.shape[0]):
                    orig_w = true_shapes[i][0]
                    orig_h = true_shapes[i][1]

                    pred_bboxes = bboxes_utils.bboxes_from_rcnn(anchors=anchors, rcnns=reg_logits[i])
                    
                    sigmoid_logits = torch.sigmoid(cl_logits[i])
                    scores, classes_predictions = torch.max(torch.sigmoid(cl_logits[i]), dim=1)

                    # small logits correspond to class background
                    mask = (sigmoid_logits < 0.3).all(dim=1)
                    classes_predictions[mask] = -1
                
                    # foreground predicrions
                    foreground_mask = classes_predictions >= 0
                    foreground_bboxes = pred_bboxes[foreground_mask]
                    foreground_classes = classes_predictions[foreground_mask]
                    foreground_scores = scores[foreground_mask]

                    # non maximum supression
                    keep1 = torchvision.ops.nms(foreground_bboxes, foreground_scores, 0.5)

                    predicted_classes = foreground_classes[keep1]
                    predicted_bboxes = foreground_bboxes[keep1]

                    # scale back bboxes and predictions (so its possible to compare with original dev test)
                    scale_x = 224 / orig_w
                    scale_y = 224 / orig_h
                    predicted_bboxes[:, [0, 2]] /= scale_x
                    predicted_bboxes[:, [1, 3]] /= scale_y

                    numbers = []
                    bboxes = []
                    for label, bbox in zip(predicted_classes, predicted_bboxes):
                        numbers.append(label)
                        bboxes.append(list(map(float, bbox)))
                    
                    all_predictions.append((numbers, bboxes))
        
        return all_predictions


    class EvaluationCallback:
        def __init__(self, test_loader,svhn, anchors):
            self.test_loader = test_loader
            self.svhn = svhn
            self.anchors = anchors


        def __call__(self, model, epoch, logs=None):
            """
            This method will be called at the end of each epoch during training.
            It generates predictions for the dev set, evaluates the model, and logs the results.
            """
            # Generate predictions and save to logdir
            all_predictions = generate_predictions(model, self.test_loader, self.anchors, False)
            accuracy = self.svhn.evaluate(self.svhn.dev, all_predictions)
            print(f"Epoch {epoch} - Accuracy: {accuracy:.4f}")
            model.train()

           
    evaluation_callback = EvaluationCallback(dev_testing,svhn, anchors)
    model.fit(train, dev=dev, epochs=args.epochs, callbacks=[evaluation_callback])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    counter = 0
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        for orig_shapes,imgs in test:
            imgs = imgs.to(torch.device('cuda'))
            with torch.no_grad():
                cl_logits, reg_logits = model(imgs)
                for i in range(reg_logits.shape[0]):
                    pred_bboxes = bboxes_utils.bboxes_from_rcnn(anchors=anchors, rcnns=reg_logits[i])
                    
                    sigmoid_logits = torch.sigmoid(cl_logits[i])
                    scores, classes_predictions = torch.max(torch.sigmoid(cl_logits[i]), dim=1)
                    mask = (sigmoid_logits < 0.3).all(dim=1)
                    classes_predictions[mask] = -1

                    # foreground anchors
                    foreground_mask = classes_predictions >= 0
                    foreground_bboxes = pred_bboxes[foreground_mask]
                    foreground_classes = classes_predictions[foreground_mask]
                    foreground_scores = scores[foreground_mask]

                    # non-maximum supression
                    keep1 = torchvision.ops.nms(foreground_bboxes, foreground_scores, 0.5)

                    predicted_classes = foreground_classes[keep1]
                    predicted_bboxes = foreground_bboxes[keep1]

                    # uncomment for predictions visualization
                    # plt.figure(figsize=(5, 5))
                    # plt.axis("off")
                    # plt.imshow(imgs[i].movedim(0, -1).numpy(force=True))
                    # for label,(top, left, bottom, right) in zip(predicted_classes.cpu(), predicted_bboxes.cpu()):
                    #     label = label.tolist() if isinstance(label, torch.Tensor) else label
                    #     plt.gca().add_patch(plt.Rectangle(
                    #         [left.cpu(), top.cpu()], right.cpu() - left.cpu(), bottom.cpu() - top.cpu(), fill=False, edgecolor=[1, 0, 1], linewidth=2))
                    #     plt.gca().text(left, top, str(label), bbox={"facecolor": [1, 0, 1], "alpha": 0.5},
                    #                 clip_box=plt.gca().clipbox, clip_on=False, ha="left", va="top")
                    #
                    # plt.savefig(f"images/prediction_{counter}.png", bbox_inches="tight", dpi=200)
                    # plt.show()
                    # plt.close()
                    counter += 1

                    # scale back
                    orig_w, orig_h = orig_shapes[i][0], orig_shapes[i][1]
                    scale_x = 224 / orig_w
                    scale_y = 224 / orig_h
                    predicted_bboxes[:, [0, 2]] /= scale_x
                    predicted_bboxes[:, [1, 3]] /= scale_y

                    output = []
                    for label, bbox in zip(predicted_classes, predicted_bboxes):
                        output += [int(label)] + list(map(float, bbox))
                    print(*output, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
