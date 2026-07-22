#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.0, type=float, help="Mask words with the given probability.")

analyses = MorphoAnalyzer("czech_pdt_analyses")
class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):

        words = example['words'].copy()
        lemmas = example['lemmas'].copy()
        tags = example['tags'].copy()

        # append word in sentence and all lemmas that come from lemmatizer for the word in example
        for word in example['words']:
            more_lemmas = analyses.get(word)
            for lemma_tag in more_lemmas:
                lemmas.append(lemma_tag.lemma)
                words.append(word)
                tags.append(lemma_tag.tag)

        # get indices to original vocabulary for words, lemmas and tags
        word_ids = self.dataset.words.string_vocab.indices(words)
        tag_ids = self.dataset.tags.string_vocab.indices(tags)
        lemmas_ids = self.dataset.lemmas.string_vocab.indices(lemmas)

        # convert to tensors
        word_ids, tag_ids, lemmas_ids = torch.tensor(word_ids,dtype=torch.long), torch.tensor(tag_ids,dtype=torch.long), torch.tensor(lemmas_ids, dtype=torch.long)
        return word_ids, words, lemmas_ids, lemmas, tag_ids

    def collate(self, batch):
        word_ids, words, lemmas_ids, lemmas, tag_ids = zip(*batch)

        # pad word_ids,lemma_ids and tag_ids with 0
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        unique_words, words_indices = self.dataset.cle_batch(words)

        # lemmas_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        # unique_lemmas, lemmas_indices = self.dataset.cle_batch(words)

        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True)
        # [B, max_sentence_len], [unique, max_word_len], [B, max_sentence_len] padded with 0
        return (word_ids, unique_words, words_indices), tag_ids


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # 1) create embedding layers for words, characters
        self._word_embedding = torch.nn.Embedding(
                                num_embeddings=len(train.words.string_vocab),
                                embedding_dim=args.we_dim
        )
        self._char_embedding = torch.nn.Embedding(
                                num_embeddings=len(train.words.char_vocab),
                                embedding_dim=args.cle_dim
        )

        # 2) define rnn bidirectional cells for words, characters
        self._char_rnn = torch.nn.GRU(
                        input_size=args.cle_dim,
                        hidden_size=args.cle_dim,
                        batch_first=True,
                        bidirectional=True
        )

        self._word_rnn = torch.nn.LSTM(input_size=2*args.cle_dim + args.we_dim,
                                            hidden_size=args.rnn_dim, 
                                            batch_first=True, 
                                            bidirectional = True
        )

        # 3) define output layer for tag classification
        num_cl = len(train.tags.string_vocab)
        self._output_layer = torch.nn.Linear(in_features=args.rnn_dim, out_features= num_cl)

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor) -> torch.Tensor:
        # create hidden state for word indices
        hidden =  self._word_embedding(word_ids) #[B, max_sent_len,we_dim]

        # character level embeddings for unique words and lemmas
        cle = self._char_embedding(unique_words) #[uni_ws, max_w_len, cle_dim]
        #lem = self._char_embedding(unique_lemmas)

        # pass character level embeddings through rnn in packed format
        lengths = torch.count_nonzero(unique_words, dim=1)
        lengths = lengths.to(torch.device('cpu'))
        packed = torch.nn.utils.rnn.pack_padded_sequence(cle, lengths= lengths, batch_first=True, enforce_sorted=False)

        # get outputs and hidden states
        packed_output, h = self._char_rnn(packed) #[2,uni_ws,B]

        # concatenate backward and forward hidden states and take embeddings that corresponds to word indices
        cle = torch.cat([h[0], h[1]], dim=-1) #[uni_ws, 2*B]
        cle = torch.nn.functional.embedding(word_indices, cle) #[B, max_sent_len, 2*B]

        # concatenate word embeddings with character-level embeddings for lemmas and words
        hidden = torch.cat([hidden, cle], dim=-1) #[B, max_sent_len, we_dim + 2*B]

        # pass whole hidden state through rnn cell for words
        lengths = torch.count_nonzero(word_ids, dim=1)
        lengths = lengths.to(torch.device('cpu'))
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, lengths= lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self._word_rnn(packed)

        # unpack the sequence, add backward and forwards outputs
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #[B, max_sent_len,2 * rnn_dim]
        sum_outputs = hidden[:, :, :self._word_rnn.hidden_size] + hidden[:, :, self._word_rnn.hidden_size:]

        # pass the logits for every word through classification layer
        hidden = self._output_layer(sum_outputs) #[B, max_sent_len,rnn_dim]
        return hidden.permute(0, 2, 1)


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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", max_sentences= args.max_sentences)

    # create train, dev, test data
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test).dataloader(batch_size=args.batch_size)

    # define the model and train the model
    model = Model(args, morpho.train)
    num_cl =  len(morpho.train.tags.string_vocab)

    model.configure(
            optimizer= torch.optim.Adam(model.parameters()),
            loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
            metrics={"accuracy": torchmetrics.Accuracy(task = "multiclass", ignore_index= morpho.PAD, num_classes= num_cl)},
            logdir=args.logdir,
        )
    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tags.txt"), "w", encoding="utf-8") as pred_vis_file:
        with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
            predictions = model.predict(test, data_with_labels=True)
            for predicted_tags, words in zip(predictions, morpho.test.words.strings):
                sent = ",".join(words)
                tags = ''
                for predicted_tag in predicted_tags[:, :len(words)].argmax(axis=0):
                    print(morpho.train.tags.string_vocab.string(predicted_tag), file=predictions_file)
                    tags += morpho.train.tags.string_vocab.string(predicted_tag) + ','
                print(sent, file=pred_vis_file)
                print(tags, file=pred_vis_file)
                print(file=pred_vis_file)
                print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
