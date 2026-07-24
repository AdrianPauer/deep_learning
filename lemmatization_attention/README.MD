# Czech Lemmatizer with Contextual Encoder and Bahdanau Attention

## Overview

This project implements a **character-level sequence-to-sequence neural lemmatizer** for Czech morphological data using **PyTorch**. The model predicts lemmas one character at a time using an encoder-decoder architecture with **Bahdanau (additive) attention**. In addition to encoding the characters of the target word, the model incorporates **sentence-level contextual information**, allowing ambiguous words to be lemmatized based on their surrounding words.

---

# Model Architecture

The model consists of four major components:

1. Character Encoder
2. Context Encoder
3. Attention-based Decoder
4. Character Prediction Layer

The complete architecture is illustrated below.

```text
                    Input Sentence
     ┌───────────────────────────────────────┐
     │ The cats are sleeping                 │
     └───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
 Character-level Encoder      Sentence Context Encoder
        │                            │
        └─────────────┬──────────────┘
                      │
            Initial Decoder State
                      │
         Attention-based GRU Decoder
                      │
          Character-by-character Output
                      │
               Predicted Lemma
```

---

# Input Representation

Each training example consists of:

* the input word represented as a sequence of characters,
* its sentence context,
* the target lemma.

Two vocabularies are maintained:

* **Character vocabulary** for character-level processing.
* **Word vocabulary** for sentence-level contextual embeddings.

---

# 1. Character Encoder

The target word is first converted into a sequence of character embeddings.

```
running

r  u  n  n  i  n  g
│  │  │  │  │  │  │
Embedding Layer
```

Each character is embedded into a vector of dimension `cle_dim` (default: **64**).

The embedded sequence is processed by a **bidirectional GRU**.

```
Character Embeddings
        │
 Bidirectional GRU
        │
Forward Hidden States
Backward Hidden States
        │
 Element-wise Sum
```

Instead of concatenating forward and backward states, the implementation **adds** them together, producing an encoder representation of size

```
(batch_size, word_length, rnn_dim)
```

These hidden vectors serve as the memory for the attention mechanism.

---

# 2. Context Encoder

To improve lemmatization accuracy, the model also learns contextual word representations.

The context encoder operates at the sentence level.

## Word Embeddings

Each word in the sentence is embedded using a trainable word embedding matrix.

```
Word IDs
   │
Embedding Layer
```

---

## Character-Level Word Embeddings (CLE)

Each unique word in the sentence is independently encoded using another bidirectional GRU over its characters.

```
Characters
     │
Character Embedding
     │
BiGRU
     │
Character-based Word Vector
```

These vectors capture morphological properties of words.

---

## Sentence Encoder

The word embedding and the character-level embedding are concatenated:

```
[word_embedding || character_embedding]
```

The resulting vectors are processed by a second bidirectional GRU operating over the entire sentence.

```
Sentence
   │
Word Representations
   │
Bidirectional GRU
   │
Context-aware Word States
```

Forward and backward states are summed to produce one contextual representation per word.

These contextual vectors initialize the decoder hidden state for every word in the sentence.

---

# 3. Bahdanau Attention

The decoder employs **Bahdanau (additive) attention**.

For every decoding step, attention scores are computed between the current decoder state and every encoder output.

```
Encoder State
      │
 Linear Projection

Decoder State
      │
 Linear Projection

        +
        │
      tanh
        │
     Linear
        │
    Softmax
        │
Attention Weights
```

The attention weights are then used to compute a weighted average of encoder outputs.

```
Encoder Outputs
        │
Weighted Sum
        │
Context Vector
```

The context vector provides the decoder with information about the most relevant characters of the input word.

---

# 4. Decoder

The decoder generates the lemma one character at a time.

Its recurrent unit is a **GRUCell** wrapped inside the custom `WithAttention` module.

At every decoding step:

1. The previous output character is embedded.
2. Bahdanau attention computes a context vector.
3. The character embedding and attention vector are concatenated.
4. The concatenated vector is processed by the GRU cell.
5. A linear output layer predicts the next character.

```
Previous Character
        │
 Character Embedding
        │
        ├──────────────┐
        │              │
        │        Attention Context
        │              │
        └──────┬───────┘
               │
         Concatenation
               │
            GRUCell
               │
         Hidden State
               │
          Linear Layer
               │
     Character Probabilities
```

---

# Training

During training, the decoder uses **teacher forcing**.

The input sequence is constructed by prepending the beginning-of-word (`<BOW>`) token and shifting the target sequence by one position.

Example:

```
Decoder Input

<BOW> r u n

Target

r u n <EOW>
```

The model is trained using **CrossEntropyLoss**, while ignoring padded characters.

Optimization is performed using the **Adam** optimizer.

---

# Inference

During prediction, the decoder no longer receives ground-truth characters.

Instead, it feeds back its own prediction into the next decoding step.

```
<BOW>
  │
  ▼
  r
  │
  ▼
  u
  │
  ▼
  n
  │
  ▼
<EOW>
```

Generation stops when either:

* the end-of-word (`<EOW>`) token is produced,
* or the maximum decoding length is reached.
---

