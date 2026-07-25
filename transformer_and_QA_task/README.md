# Transformer Encoder for Morphological Tagging

This project implements a simplified **Transformer Encoder** from scratch using PyTorch.  
The model is trained for **morphological tagging** (sequence labeling), where each word in a sentence is assigned a grammatical tag.

The implementation is inspired by the Transformer architecture introduced in paper:

>  *Attention Is All You Need* (2017)

## Dataset
The model is trained on the **Czech CAC morphological dataset**.

## Architecture
The model consists of:

1. **Word Embedding Layer**
   - Converts input word IDs into dense vector representations.

2. **Sinusoidal Positional Embeddings**
   - Adds information about word order because self-attention alone is position-independent.
   - Uses sine and cosine functions with different frequencies.

3. **Transformer Encoder**
   - Multiple stacked encoder layers containing:
     - Multi-head self-attention
     - Feed-forward neural network (FFN)
     - Layer normalization
     - Residual connections
     - Dropout

4. **Classification Layer**
   - Maps contextual word representations to morphological tag probabilities.

![img.png](img.png)
## Self-Attention

The self-attention mechanism computes:

\[ Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V \]

where:

- Query (Q), Key (K), and Value (V) projections are learned using weight matrices.
- Multiple attention heads allow the model to capture different relationships between words.
- The output of all heads is concatenated and projected using the output matrix \(W_O\).

## Implementation Details

The project includes:

- Xavier uniform initialization for attention weight matrices.
- Multi-head self-attention implemented manually.
- Padding masks to ignore padded tokens.
- Residual connections around attention and FFN blocks.
- Adam optimizer for training.
- Cross-entropy loss for tag prediction.


## Predictions on test data
```
sentence: Zásady správné a úspěšné úpravy .
predicted tags: NOUN,ADJ,CCONJ,ADJ,NOUN,PUNCT
gold tags: NOUN,ADJ,CCONJ,ADJ,NOUN,PUNCT

sentence: Teoretické skončení pracovní neschopnosti je tehdy , když pracovník je schopen vykonávat původní zaměstnání .
predicted tags: ADJ,NOUN,ADJ,NOUN,AUX,ADV,PUNCT,SCONJ,NOUN,AUX,ADJ,VERB,ADJ,NOUN,PUNCT
gold tags: ADJ,NOUN,ADJ,NOUN,VERB,ADV,PUNCT,SCONJ,NOUN,AUX,ADJ,VERB,ADJ,NOUN,PUNCT

sentence: Tak co bude , mládenci .
predicted tags: ADV,PRON,AUX,PUNCT,NOUN,PUNCT
gold tags: ADV,PRON,VERB,PUNCT,NOUN,PUNCT

sentence: Nezřídka se uplatňuje názor , že na průběh a výsledky empirického výzkumu nemají podstatný vliv .
predicted tags: ADV,PRON,VERB,NOUN,PUNCT,SCONJ,ADP,NOUN,CCONJ,NOUN,ADJ,NOUN,VERB,ADJ,NOUN,PUNCT
gold tags: ADV,PRON,VERB,NOUN,PUNCT,SCONJ,ADP,NOUN,CCONJ,NOUN,ADJ,NOUN,VERB,ADJ,NOUN,PUNCT

sentence: Nemálo autorů se domnívá , že to jsou otázky sice důležité , ale jen pro filozofické spory .
predicted tags: ADV,NOUN,PRON,VERB,PUNCT,SCONJ,DET,AUX,NOUN,ADV,ADJ,PUNCT,CCONJ,PART,ADP,ADJ,NOUN,PUNCT
gold tags: DET,NOUN,PRON,VERB,PUNCT,SCONJ,DET,AUX,NOUN,ADV,ADJ,PUNCT,CCONJ,PART,ADP,ADJ,NOUN,PUNCT
```




