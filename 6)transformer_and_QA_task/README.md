# Transformer Encoder and Question answering task
## 1) Transformer encoder for morphological tagging
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

# 2) Reading Comprehension with Robeczech

This project implements an extractive **Question Answering (QA)** model for Czech reading comprehension using the **RobeCzech** transformer encoder. The model predicts the start and end token positions of the answer span within a given context.

## Features
* Fine-tuning of the pretrained **ufal/robeczech-base** model
* Token-level start/end span prediction
* Dynamic batching and padding
* Cosine learning rate scheduler with warm-up

## Model Architecture

The model consists of:

* **Backbone:** `ufal/robeczech-base`
* **Classification Head:** Linear layer projecting the hidden representation (768) to two logits:

  * start position
  * end position

```
Context + Question
        │
        ▼
  RobeCzech Encoder
        │
        ▼
 Linear Classification Head
        │
        ├── Start logits
        └── End logits
```

## Dataset

The project uses the `ReadingComprehensionDataset` provided by the `npfl138` package.

Each example contains:

* a context paragraph
* one or more questions
* answer spans (training/dev)
* no answers for the test set

During preprocessing:

* the context and question are tokenized together
* the context is truncated to a maximum sequence length of 512 tokens
* character-level answer spans are converted to token indices using `char_to_token()`


## Prediction

For each question:

1. The model predicts start and end logits.
2. The predicted token span is converted back into the original text answer.


The `npfl138` package is provided as part of the course environment.

## Notes

* The model predicts answer spans rather than generating free-form answers.
* Contexts longer than 512 tokens are truncated.
* Questions whose answer span falls outside the truncated context are skipped during training.

## Sample from predicitons on test data


| Question                                                                         | Predicted answer                    | Context                                                                                                    |
|----------------------------------------------------------------------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------|
| Která republika si udržela kontrolu nad Íránem?                                  | **Islámská republika**              | "... **Islámská republika** si také udržela moc v Íránu navzdory..."                                       |
| Kdo prohlásil, že chce, aby Izrael zmizel?                                       | **prezidenta Mahmúda Ahmadínežáda** | "... odporu prezidenta **Mahmúda Ahmadínežáda** vůči Spojeným státům a jeho výzvě, aby Izrael zmizel."     |
| Co používal Západ k ospravedlnění kontroly nad východními územími?               | **Diskuse Orientalismu**            | "... **Diskuse Orientalismu** proto sloužila jako ideologické ospravedlnění ... "                          |
| Přístav Long Beach patří do které oblasti Kalifornie?                            | **Jižní Kalifornie**                | "... **Jižní Kalifornie** je také domovem přístavu Los Angeles..., přilehlého přístavu Long Beach, ..."    |
| Jaký je druhý nejvytíženější kontejnerový přístav ve Spojených státech?          | **přístavu Long Beach**             | "...přilehlého **přístavu Long Beach**, druhého nejrušnějšího kontejnerového přístavu Spojených států,..." |
| Jaké jsou dva typy fagocytů, které cestují tělem, aby našly napadající patogeny? | **Neutrofily a makrofágy**          | "... **Neutrofily a makrofágy** jsou fagocyty, které putují celým tělem ve snaze napadnout patogeny. "     |
| Jaký je nejhojnější druh fagocytů?                                               | **Neutrofily**                      | "... **Neutrofily** se běžně vyskytují v krevním oběhu..."                                                 |
| Kdy Francie převzala kontrolu nad Alžírskem?                                     | **1830**                            | "... převzala kontrolu nad Alžírskem v roce **1830**, ale po roce..."                                      |
| Kdy začala Francie vážně budovat své globální impérium?                          | **1850**                            | "... ale po roce **1850** začala vážně budovat své celosvětové impérium a soustřeďovala se hlavně..."      |

