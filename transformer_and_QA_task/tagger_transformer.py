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
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--transformer_dropout", default=0.2, type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=5, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=46, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")

class Model(npfl138.TrainableModule):
    # define feed forward model
    class FFN(torch.nn.Module):
        def __init__(self, dim: int, expansion: int) -> None:
            super().__init__()
            self._model = torch.nn.Sequential(
                torch.nn.Linear(in_features= dim, out_features= dim * expansion),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features= dim * expansion, out_features=dim)
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self._model(inputs)

    class SelfAttention(torch.nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            #  define weight matrices for self attention
            self.dim, self.heads = dim, heads

            self.W_Q, self.W_K, self.W_V = torch.nn.Parameter(torch.empty(dim,dim)), torch.nn.Parameter(torch.empty(dim,dim)), torch.nn.Parameter(torch.empty(dim,dim))
            self.W_O = torch.nn.Parameter(torch.empty(dim,dim))

            torch.nn.init.xavier_uniform_(self.W_Q)
            torch.nn.init.xavier_uniform_(self.W_K)
            torch.nn.init.xavier_uniform_(self.W_V)
            torch.nn.init.xavier_uniform_(self.W_O)

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            B = inputs.shape[0]
            max_sentence_len = inputs.shape[1]

            # compute Q,K,V matrices
            Q = (inputs @ self.W_Q).reshape([B, max_sentence_len, self.heads, self.dim // self.heads]).permute([0,2,1,3])
            K = (inputs @ self.W_K).reshape([B, max_sentence_len, self.heads, self.dim // self.heads]).permute([0,2,1,3])
            V = (inputs @ self.W_V).reshape([B, max_sentence_len, self.heads, self.dim // self.heads]).permute([0,2,1,3]) #[B,max_len_sent, heads, dim/heads]

            # Continue by computing the self-attention weights as Q @ K^T, normalizing by the square root of `dim // heads`.
            self_attention = Q @ K.mT / torch.sqrt(torch.tensor(self.dim // self.heads)) #[B,heads,max_len_sent, max_len_sent]
            # Expand the mask to the shape [B,heads, max_len_sent, max_len_sent ]
            mask = mask[:, None, None, :] * mask[:, None, :, None]
            mask = mask.expand(-1,self.heads,-1,-1)
            self_attention[~mask] = -1e9
            
            probs = torch.softmax(self_attention, dim = -1) #[B,heads, max_len_sent, max_len_sent ]

            weighted_comb = probs @ V #[B,heads, max_len_sent, we_dim // heads]
            weighted_comb = weighted_comb.permute([0, 2, 1, 3])
            weighted_comb = weighted_comb.reshape([B, max_sentence_len, self.dim ])

            return weighted_comb @ self.W_O

    class PositionalEmbedding(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # Sinusoidal and cosinusoidal positional embeddings.
            B,S,dim = inputs.shape
            pos = torch.arange(S, device=inputs.device).unsqueeze(1)  # [S, 1]
            i = torch.arange(dim // 2, device=inputs.device).unsqueeze(0)  # [1, dim/2]

            angle_rates = pos / (10000 ** (2 * i / dim))  # [S, dim/2]
            pos_emb = torch.zeros(S, dim, device=inputs.device)  # [S, dim]

            pos_emb[:, :dim // 2] = torch.sin(angle_rates)
            pos_emb[:, dim // 2:] = torch.cos(angle_rates)

            pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1)  # [B, S, dim]
            return pos_emb

    class Transformer(torch.nn.Module):
        def __init__(self, layers: int, dim: int, expansion: int, heads: int, dropout: float) -> None:
            super().__init__()
            self.layers = layers

            self.pos_emebd = Model.PositionalEmbedding()
            self.attention_layers = torch.nn.ModuleList()

            # stack the transformer layers
            for i in range(layers):
                self.attention_layers.append( torch.nn.ModuleList( [
                                            torch.nn.LayerNorm(dim),
                                            Model.SelfAttention(
                                                dim = dim,
                                                heads = heads),
                                            torch.nn.Dropout(dropout),
                                            torch.nn.LayerNorm(dim),
                                            Model.FFN(dim = dim, expansion=expansion),
                                            torch.nn.Dropout(dropout)])
                                            )
            


        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # First compute the positional embeddings.
            pos_embed = self.pos_emebd(inputs) #[B,max_len_sent,we_dim]
            hidden = pos_embed + inputs #[B,max_len_sent,we_dim]

            for module in self.attention_layers:
                # transformer block 1) -> layer_norm -> self_attention -> Droupout -> ADD
                att_norm,att, att_dropout, ffn_norm, ffn, ffn_dropout = module
                x = att_norm(hidden)
                x = att(x, mask)
                x = att_dropout(x)

                hidden = hidden + x
                # transformer block 2) -> layer norm -> FFN -> Droupout -> ADD
                x = ffn_norm(hidden)
                x = ffn(x)
                x = ffn_dropout(x)

                hidden = hidden + x

            return hidden

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        # define whole model
        vocab_size = len(train.words.string_vocab)
        num_cl =  len(train.tags.string_vocab)

        self._word_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=args.we_dim, padding_idx=MorphoDataset.PAD)
        self._transformer = Model.Transformer(dim=args.we_dim,
                                              layers=args.transformer_layers,
                                              expansion=args.transformer_expansion,
                                              heads=args.transformer_heads,
                                              dropout=args.transformer_dropout)
        self._output_layer = torch.nn.Linear(in_features=args.we_dim, out_features= num_cl)

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._word_embedding(word_ids) #[B,max_len_sent, we_dim]
        hidden = self._transformer(hidden, mask = word_ids != MorphoDataset.PAD ) #[B,max_len_sent, we_dim]
        hidden = self._output_layer(hidden) #[B,max_len_sent, we_dim]
        return hidden.permute([0,2,1]) #[B, we_dim,max_len_sent]


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # return sequences as sequences of indices to vocabulary
        word_ids = self.dataset.words.string_vocab.indices(example['words'])
        tag_ids = self.dataset.tags.string_vocab.indices(example['tags'])
        return torch.tensor(word_ids,dtype=torch.long), torch.tensor(tag_ids,dtype=torch.long)

    def collate(self, batch):
        # pad word and tag ids
        word_ids, tag_ids = zip(*batch)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True) #[B, max_len_sent]
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True) #[B, max_len_sent]
        return word_ids, tag_ids


def main(args: argparse.Namespace) -> dict[str, float]:
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)
    num_cl =  len(morpho.train.tags.string_vocab)

    model.configure(
        optimizer= torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics={"accuracy": torchmetrics.Accuracy(task = "multiclass", ignore_index= morpho.PAD, num_classes= num_cl)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # generate tags on test set
    with open('predicitons_test.txt', 'w', encoding='utf-8') as f:
        for batch in test:
            word_ids,tag_ids = batch
            mask=word_ids != MorphoDataset.PAD
            logits = model(word_ids.to('cuda'))
            logits = logits.permute([0,2,1])
            preds = logits.argmax(-1)
            for i in range(len(batch)):
                sent = preds[i][mask[i]]
                tags_sent = tag_ids[i][mask[i]]
                print('sentence: ' + ' '.join(morpho.test.words.string_vocab.strings(word_ids[i][mask[i]])), file=f)
                print('predicted tags: ' + ','.join(morpho.test.tags.string_vocab.strings(sent)), file=f)
                print('gold tags: ' + ','.join(morpho.test.tags.string_vocab.strings(tags_sent)), file=f)
                print(file=f)


    # Return development and training losses for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if "loss" in metric}

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
