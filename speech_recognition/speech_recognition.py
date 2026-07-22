#!/usr/bin/env python3

import argparse
import datetime
import os
import re

import torch
import torchaudio.models.decoder

import npfl138
from npfl138.datasets.common_voice_cs import CommonVoiceCs

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--hidden_dim", default=512, type=int, help="Maximum number of threads to use.")
parser.add_argument("--beam_size", default=5, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.5, type=float, help="Maximum number of threads to use.")
parser.add_argument("--rnn_layers", default=4, type=int, help="Maximum number of threads to use.")

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()

        self.counter = 0
        self._args = args
        self.num_classes = len(CommonVoiceCs.LETTER_NAMES)
        self.blank_index = CommonVoiceCs.LETTER_NAMES.index("[PAD]")

        # define the model
        self._dropout_layer = torch.nn.Dropout(args.dropout)
        # stacked bidirectional LSTM celss
        self._rnn = torch.nn.ModuleList(
            [torch.nn.LSTM(CommonVoiceCs.MFCC_DIM, args.hidden_dim, batch_first=True, bidirectional=True)]
        )
        self._rnn.extend([torch.nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True, bidirectional=True) for _ in range(args.rnn_layers - 1)])
        # output layer
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(self._args.hidden_dim, self._args.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(self._args.hidden_dim, self.num_classes),
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:

        # pass the padded time sequences through RNN and classification layer
        hidden = torch.nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # apply each rnn layer, sum the forward, backward logits, apply dropout and add to original data
        for i, rnn_layer in enumerate(self._rnn):
            residual  = hidden
            hidden,_ = rnn_layer(hidden)
            # extract forward, backward logits
            forward, backward = torch.chunk(hidden.data, 2, dim=-1)
            hidden = self._dropout_layer(forward + backward)
            # in first rnn layer output and input dimensions does not match
            if i > 0:hidden += residual.data
            hidden = torch.nn.utils.rnn.PackedSequence(hidden, *residual[1:])

        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)
        # apply classification layer
        logits = self.classifier(hidden)

        return logits

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        # get lengths of gold sentences and prepare gold data for ctc loss
        mask = y_true != self.blank_index
        target_lengths = mask.sum(dim=1)
        targets = y_true[mask]

        # CTC loss expects [T,B,C] and log-probabilities
        y_pred = y_pred.transpose(0, 1).log_softmax(dim=-1)

        return self.loss(y_pred, targets, input_lengths, target_lengths)

    def ctc_decoding(self, y_pred: torch.Tensor,inputs: torch.Tensor, input_lengths: torch.Tensor) -> list[torch.Tensor]:
        # define decoder with beam search
        decoder = torchaudio.models.decoder.cuda_ctc_decoder(
            tokens=CommonVoiceCs.LETTER_NAMES,
            nbest=1,
            beam_size=self._args.beam_size,
        )
        # compute probabilities from logits
        y_pred = y_pred.log_softmax(dim=-1)

        # use CTC decoder to get result sentences
        results = decoder(y_pred, input_lengths.to(torch.int32))
        pred_sequences = [torch.tensor(result[0].tokens, dtype=torch.int32) for result in results]

        return pred_sequences

    def compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor,inputs: torch.Tensor, input_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        # only when `self.training==False` to speed up training.
        if self.training:
            return {}

        # get predictions and update the metrics
        predictions = self.ctc_decoding(y_pred, inputs, input_lengths)
        self.metrics["edit_distance"].update(predictions, y_true)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            for sentence in batch:
                print("".join(CommonVoiceCs.LETTER_NAMES[char] for char in sentence))
            # if as_numpy:
            #     batch = [example.numpy(force=True) for example in batch]
            return batch

class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        letter_ids = torch.LongTensor([CommonVoiceCs.LETTER_NAMES.index(letter) for letter in example["sentence"]])
        mfccs = example["mfccs"]
        return mfccs, letter_ids

    def collate(self, batch):
        mfccs, tag_ids = zip(*batch)
        mfc_lengths = torch.LongTensor([len(seq) for seq in mfccs])

        # pad train and gold data by 0
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=0) #[B, max_sent_len]
        mfccs = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True, padding_value=0) #[B,max_len_time,13]

        return (mfccs, mfc_lengths), tag_ids


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

    # Load the data.
    common_voice = CommonVoiceCs()

    train = TrainableDataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(common_voice.dev).dataloader(args.batch_size)
    test = TrainableDataset(common_voice.test).dataloader(args.batch_size)

    model = Model(args, train=train)

    model.configure(
        optimizer= torch.optim.Adam(model.parameters()),
        loss=torch.nn.CTCLoss(blank=0, zero_infinity=True),
        metrics={
            "edit_distance": common_voice.EditDistanceMetric(ignore_index=0),
        },
        logdir=args.logdir,
    )

    model.fit(train, dev = dev, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition_dev.txt"), "w", encoding="utf-8") as predictions_file:
        predictions_dev = model.predict(dev, data_with_labels=True)
        for sentence in predictions_dev:
            print("".join(CommonVoiceCs.LETTER_NAMES[char] for char in sentence), file=predictions_file)

    # Generate predictions on test set
    # with open(os.path.join(args.logdir, "speech_recognition_test.txt"), "w", encoding="utf-8") as predictions_file:
    #     predictions_dev = model.predict(test, data_with_labels=True)
    #     for sentence in predictions_dev:
    #         print("".join(CommonVoiceCs.LETTER_NAMES[char] for char in sentence), file=predictions_file)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
