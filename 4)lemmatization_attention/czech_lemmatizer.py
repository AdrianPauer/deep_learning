#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=10, type=int, help="Show results every given batch.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell
        self._project_encoder_layer = torch.nn.Linear(in_features= cell.hidden_size, out_features= attention_dim)
        self._project_decoder_layer = torch.nn.Linear(in_features= cell.hidden_size, out_features= attention_dim)
        self._output_layer = torch.nn.Linear(in_features= attention_dim, out_features= 1)

    def setup_memory(self, encoded):
        # save the encoder states and projection for bahdannau attention weights computation
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        # project decoder states and sum with projected encoder states
        decoded_projected = self._project_decoder_layer(states) #[max_w, attention_dim]
        sum_projected = decoded_projected[:, torch.newaxis, :] + self._encoded_projected #[max_w, chars, cle_dim]

        result = self._output_layer(torch.tanh(sum_projected)) #[max_w, 13]
        weights = torch.softmax(result, dim=1) #[max_w,13.1]

        # get context vector as a weighted sum of vectors corresponding to sequence parts, concatenate with inputs
        attention = (self._encoded * weights).sum(dim = 1)
        inputs_attention = torch.cat([inputs, attention], dim=1)

        # pass through rnn initialized with states
        return self._cell(inputs_attention, states)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset, dev) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab
        self._train = train
        self._dev = dev

        # Define source embedding layer and rnn cell
        self._source_embedding = torch.nn.Embedding(num_embeddings=len(self._source_vocab),
                                                    embedding_dim=args.cle_dim,
                                                    padding_idx=MorphoDataset.PAD
                                                    )
        
        self._source_rnn = torch.nn.GRU(input_size=args.cle_dim,
                                        hidden_size=args.rnn_dim,
                                        batch_first=True,
                                        bidirectional = True
                                        )

        # Define target rnn cell with Bahdanau attention mechanism and output layer
        self._target_rnn_cell = WithAttention(cell= torch.nn.GRUCell(
                                                     input_size=args.cle_dim + args.rnn_dim,
                                                     hidden_size= args.rnn_dim
                                                     ),
                                              attention_dim= args.rnn_dim
                                              )
        self._target_output_layer = torch.nn.Linear(in_features=args.rnn_dim, out_features=len(self._target_vocab))
        self._target_embedding =  torch.nn.Embedding(num_embeddings=len(self._target_vocab),
                                                    embedding_dim=args.cle_dim,
                                                    padding_idx=MorphoDataset.PAD
                                                     )

        # define embedding layer for words and rnn cells for words and characters
        self._word_embedding = torch.nn.Embedding(
            num_embeddings=len(train.words.string_vocab),
            embedding_dim=args.cle_dim,
            padding_idx=train.words.string_vocab.PAD
        )
        
        self._char_rnn = torch.nn.GRU(
            input_size=args.cle_dim,
            hidden_size=args.cle_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self._word_rnn = torch.nn.GRU(
            input_size=args.cle_dim + 2*args.cle_dim,
            hidden_size=args.rnn_dim,
            batch_first=True,
            bidirectional = True
        )
        

    def forward(self, words: torch.Tensor,  word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded,states = self.encoder(words, word_ids, unique_words, word_indices)
        if targets is not None:
            return self.decoder_training(encoded, targets, states)
        else:
            return self.decoder_prediction(encoded, states, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor) -> torch.Tensor:
        # embed the words and pass through source rnn
        hidden = self._source_embedding(words) #[all_words, chars, cle]
        lengths = torch.count_nonzero(words, dim=1).to('cpu')
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, lengths= lengths, batch_first=True, enforce_sorted=False)
        hidden,_ = self._source_rnn(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True) #[B,chars, 2 * rnn_dim]
        # sum backward and forward hidden states and get encoder states
        H = outputs.shape[2] // 2
        outputs = outputs[:, :, :H] + outputs[:, :, H:] #[B,chars, rnn_dim]

        # pass the sentences through embedding and rnn layers
        hidden =  self._word_embedding(word_ids) #[B,max_len_sent, cle_dim]
        cle = self._source_embedding(unique_words) #[uni_w, chars,cle_dim]
        # pass cle for unique words through char_rnn, concatenate froward and backward outputs
        lengths = torch.count_nonzero(unique_words, dim=1).to('cpu')
        packed =   torch.nn.utils.rnn.pack_padded_sequence(cle, lengths= lengths, batch_first=True, enforce_sorted=False)
        packed_output, h = self._char_rnn(packed)
        cle = torch.cat([h[0], h[1]], dim=-1) #[uni_w,chars,2*cle_dim]
        # TODO: insert CLE into sequences
        cle = torch.nn.functional.embedding(word_indices, cle) #[B,max_len_sent, 2*cle_dim]

        # concatenate cle and words embeddings, pass through word rnn
        hidden = torch.cat([hidden, cle], dim=-1) # [B, max_len_sent, 3*cle_dim]
        lengths = torch.count_nonzero(word_ids, dim=1).to('cpu')
        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, lengths= lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self._word_rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # get states by summing backward and forward outputs
        states = hidden[:, :, :self._word_rnn.hidden_size] + hidden[:, :, self._word_rnn.hidden_size:] #[B, max_len_sent, rnn_dim]
        sentence_mask = word_ids != self._train.words.string_vocab.PAD

        # return flattened states corresponding to non-padding word in sentences
        B, T, H = states.shape
        flattened_hidden = states.view(B * T, H)
        flattened_mask = sentence_mask.view(B * T)
        states = flattened_hidden[flattened_mask]

        return outputs, states

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor, init_states: torch.Tensor) -> torch.Tensor:
        # construct inputs to decoder as a vector of BOW characters and pass encoder states to decoder
        B = encoded.shape[0]
        decoder_inputs = torch.cat([torch.full((B,1), MorphoDataset.BOW).to('cuda'), targets[:,:-1] ], dim=1) #[max_w, chars]
        self._target_rnn_cell.setup_memory(encoded)

        embed_t = self._target_embedding(decoder_inputs) #[max_w,chars,cle]
        states = init_states
        outputs = []
        # set initial states of decoder rnn to states and decode the input
        for i in range(embed_t.shape[1]):
            input_t = embed_t[:, i] 
            states = self._target_rnn_cell(input_t, states) 
            outputs.append(states)
        
        outputs = torch.stack(outputs, dim=1) #[max_w, chars, cle]
        logits = self._target_output_layer(outputs) #[max_w,chars,vocab_size]

        return logits.permute(0, 2, 1)

    def decoder_prediction(self, encoded: torch.Tensor, init_states: torch.Tensor, max_length: int) -> torch.Tensor:
        # set up the decoder state to encoder output and autoregressively predict sequence.
        batch_size = encoded.shape[0]
        self._target_rnn_cell.setup_memory(encoded)
        index = 0
        inputs = torch.full([batch_size], MorphoDataset.BOW).to('cuda')
        states = init_states
        results = []
        result_lengths = torch.full([batch_size], max_length).to('cuda')

        while index < max_length and torch.any(result_lengths == max_length):
            # embed the inputs and pass through rnn with attention
            embed_i = self._target_embedding(inputs) #[B, cle_dim]
            hidden = self._target_rnn_cell(embed_i, states)
            states = hidden

            predictions = self._target_output_layer(hidden) #[B,vocab]
            predictions = predictions.argmax(dim=-1)\

            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if self.training:  # In training regime, convert logits to most likely predictions.
            y_pred = y_pred.argmax(dim=-2)
        # Compare the lemmas with the predictions using exact match accuracy.
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        self.metrics["accuracy"].update(torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1))
        return {name: metric.compute() for name, metric in self.metrics.items()}  # Return all metrics.

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs, as_numpy=False):
        with torch.no_grad():
            batch = self.forward(*xs)
            # Trim the predictions at the first EOW
            batch = [lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0] for lemma in batch]
            return batch
            return [lemma.numpy(force=True) for lemma in batch] if as_numpy else batch


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        # transform words into vocab ids
        word_ids = torch.LongTensor([self._dataset.words.string_vocab.index(w) for w in example["words"]])
        return example['words'], example['lemmas'], word_ids

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples generated by `transform`.
        words, lemmas, word_ids = zip(*batch)

        # pad the word id sequences with 0 and get unique words and indices from sequences
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        unique_words, words_indices = self.dataset.cle_batch(words) #[B,unique_w], [B, max_len_sent]

        # map words and lemmas into characters indices
        chars_words = [ torch.tensor(self.dataset.words.char_vocab.indices(word)) for sentence in words for word in sentence]
        words = torch.nn.utils.rnn.pad_sequence(chars_words, batch_first=True) #[all words, max_len_w]

        # additionally, append `MorphoDataset.EOW` to the end of each lemma.
        chars_lemmas = [torch.tensor(self.dataset.lemmas.char_vocab.indices(word) + [MorphoDataset.EOW]) for sentence in lemmas for word in sentence]
        lemmas = torch.nn.utils.rnn.pad_sequence(chars_lemmas, batch_first=True)#[all_words, max_len_l]

        # Return a pair (inputs, targets), where
        # - the inputs are words during inference and (words, lemmas) pair during training;
        # - the targets are lemmas.

        if self._training:
            inputs = (words, word_ids, unique_words, words_indices, lemmas)
        else:
            inputs = words, word_ids, unique_words, words_indices

        return inputs, lemmas 
    
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
    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)

    train = TrainableDataset(morpho.train, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, training=False).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test, training=False).dataloader(batch_size=args.batch_size)

    model = Model(args, morpho.train, morpho.dev)
    
    model.configure(
        optimizer= torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics={"accuracy": torchmetrics.MeanMetric()},
        logdir=args.logdir,
    )

    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = iter(model.predict(test, data_with_labels=True))
        for sentence in morpho.test.words.strings:
            for word in sentence:
                lemma = next(predictions)
                print("".join(morpho.test.lemmas.char_vocab.strings(lemma)), file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
