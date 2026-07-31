#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics
import transformers
import npfl138

from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset
from transformers import get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--hidden_dim", default=1024, type=int)
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
parser.add_argument("--threads", default=4, type=int, help="Number of threads.")


class Model(npfl138.TrainableModule):
    def __init__(self,robeczech: transformers.PreTrainedModel) -> None:
        super().__init__()

        self._robeczech = robeczech
        self._classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 2)
        )

    def forward(self, inputs: torch.Tensor, mask) -> torch.Tensor:
        hidden_states = self._robeczech(inputs, attention_mask=mask).last_hidden_state
        logits = self._classifier(hidden_states)
        start_logits, end_logits = logits[:,:, 0], logits[:,:, 1]

        # fill padding tokens with -inf
        mask = mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits = end_logits.masked_fill(~mask, float('-inf'))

        return start_logits, end_logits

    def compute_loss(self, y_pred, y, *xs) -> torch.Tensor:
        start_loss = self.loss(y_pred[0], y[:,0])
        end_loss = self.loss(y_pred[1], y[:,1])

        # compute averaged crossentropy loss
        return (start_loss + end_loss) / 2

    def compute_metrics(self, y_pred, y, *xs):
        start_logits, end_logits = y_pred
        start_targets, end_targets = y[:, 0], y[:, 1]

        # Compute predicted start/end indices
        start_preds = start_logits.argmax(dim=-1)
        end_preds = end_logits.argmax(dim=-1)

        self.metrics["start_acc"].update(start_preds, start_targets)
        self.metrics["end_acc"].update(end_preds, end_targets)

        return {
            "start_acc": self.metrics["start_acc"].compute(),
            "end_acc": self.metrics["end_acc"].compute(),
        }


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: ReadingComprehensionDataset.Dataset, tokenizer, test) -> None:
        super().__init__(dataset.paragraphs)
        self.tokenizer = tokenizer
        self.test = test

    def transform(self, example):
        tokenized_examples = []
        start_ends = []

        for qa in example["qas"]:
            question = qa["question"]
            context = example["context"]

            tokenized = self.tokenizer(
                context,
                question,
                max_length=512,
                truncation='only_first',
                padding="max_length",
                return_offsets_mapping=True
            )

            if not self.test:
                if len(qa["answers"]):
                    answer = qa["answers"][0]["text"]
                    start_char = qa["answers"][0]["start"]
                    end_char = start_char + len(answer) - 1
                    # create gold data as (start token, end token)
                    start_token, end_token = tokenized.char_to_token(start_char), tokenized.char_to_token(end_char)

                    # if start,end tokens are out of lenght 512
                    if end_token == None or start_token == None:
                        continue

                    start_ends.append((start_token, end_token))

            tokenized_examples.append(tokenized)

        return tokenized_examples, start_ends

    def collate(self, batch):
        # create batch from tokenized examples and gold labels
        all_encodings = []
        all_labels = []
        for tokenized_list, label_list in batch:
            all_encodings.extend(tokenized_list)
            all_labels.extend(label_list)

        # padd the batch with 1s
        batch_enc = self.tokenizer.pad(
            all_encodings,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        # return ctx + q tokens [num_answers_across_the_batch, 512] , attention mask, labels [num_answers_across_the_batch, 2]
        if self.test:
            return batch_enc["input_ids"], batch_enc["attention_mask"]
        else:
            labels = torch.tensor(all_labels, dtype=torch.long)
            return (batch_enc["input_ids"], batch_enc["attention_mask"]), labels

def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the pre-trained RobeCzech model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data
    dataset = ReadingComprehensionDataset()
    train = TrainableDataset(dataset.train, tokenizer, False).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(dataset.dev, tokenizer, False).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(dataset.test, tokenizer, True).dataloader(batch_size=args.batch_size)

    model = Model(robeczech).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define warmup steps to scheduler
    total_steps = args.epochs * len(train)
    warmup_steps = int(0.15 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
        loss=torch.nn.CrossEntropyLoss(),
        logdir=args.logdir,
        metrics={"start_acc": torchmetrics.classification.MulticlassAccuracy(num_classes=512), "end_acc": torchmetrics.classification.MulticlassAccuracy(num_classes=512)},
    )

    def predict(data, name):
        answers = []
        batches = iter(data)
        for batch in batches:
            if name == "dev":
                (input_ids, attention_mask), _ = batch
            else:
                input_ids, attention_mask = batch
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            answer_starts, answer_ends = model(input_ids, attention_mask)

            start_index = answer_starts.argmax(-1)
            end_index = answer_ends.argmax(-1)
            for i in range(len(end_index)):
                if start_index[i] < end_index[i]:
                    answers.append(tokenizer.decode(input_ids[i][start_index[i]:end_index[i]]))
                else:
                    answers.append("")
        return answers

    class EvaluationCallback:
        def __init__(self, comprehension_dataset):
            self.comprehension_dataset = comprehension_dataset

        def __call__(self, model, epoch, logs=None):
            """
            This method will be called at the end of each epoch during training.
            It generates predictions for the test set, evaluates the model, and logs the results.
            """
            model.eval()
            all_predictions = predict(dev, "dev")
            accuracy = self.comprehension_dataset.evaluate(self.comprehension_dataset.dev, all_predictions)
            print(f"Epoch {epoch} - Accuracy: {accuracy:.4f}")
            model.train()

    eval_callback = EvaluationCallback(dataset)
    #model.fit(train, dev=dev, epochs=args.epochs, callbacks=[eval_callback])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    # with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
    #     model.eval()
    #     with torch.no_grad():
    #         predictions = predict(test,"test")
    #     for answer in predictions:
    #         print(answer, file=predictions_file)

    with open("reading_comprehension_results.txt", "w", encoding="utf-8") as f:
        with open("reading_comprehension.txt", "r", encoding="utf-8") as predictions_file:
            data = predictions_file.readlines()
            counter = 0
            for i in range(len(dataset.test.paragraphs)):
                example = dataset.test.paragraphs[i]
                context = example["context"]
                for qas in example["qas"]:
                    question = qas["question"]
                    print( '**context:**'+context, file = f)
                    print('**question:**' + question, file = f)
                    print('**answer**:',data[counter], file = f)
                    counter += 1


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
