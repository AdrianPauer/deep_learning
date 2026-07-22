# Speech Recognition (CommonVoice)

This project contains an RNN-based speech recognition experiment using the CommonVoice dataset. The pipeline uses MFCC features, stacked bidirectional LSTM residual blocks, CTC loss for training, and beam-search decoding for prediction.

## Dataset

- Source: Mozilla CommonVoice (development sample: https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/demos/common_voice_cs_dev.html)
- Input representation: MFCC features (13 coefficients per frame) extracted from audio and organized as time-step sequences.
- Preprocessing: sequences are padded and collated into batches. Targets are character index sequences derived from ground-truth transcripts.

## Model

- Encoder: stacked residual blocks of bidirectional LSTM layers with dropout for regularization.
- Classification head: two fully connected layers with ReLU and dropout that project encoder outputs to character logits.
- Loss: Connectionist Temporal Classification (CTC) via torch.nn.CTCLoss.

## Training pipeline

1. Extract MFCC features and pad sequences.
2. Forward the batch through the RNN residual encoder (e.g., 4 blocks).
3. Apply the classification head to produce logits over the character vocabulary.
4. Compute CTC loss and update model parameters.
5. Decode predictions using beam-search (e.g., torchaudio's CTC decoder or another implementation).

## Predicitons on development set 


| # | Model Prediction | Gold Reference |
|---|------------------|----------------|
| 1 | vlédi jezeru znemotá pěší toristikájí zden horských olechnígi | v létě je zde rozvinutá pěší turistika a jízda na horských kolech |
| 2 | vosilník pavivaso neobly lidal ne z doje nriie | fosilní paliva jsou neobnovitelné zdroje energie |
| 3 | ttejně tad nognářké katní | stejně tak novinářské kachny |
| 4 | v součestnosti jsou ale a hilunicvečení v neji různějších porobarkele vilmi topuléní | v současnosti jsou ale aerobní cvičení v nejrůznějších podobách dále velmi populární |
| 5 | taky ježtě nemále všechni ceny dotomborií | taky ještě nemáme všechny ceny do tomboly |
| 6 | onečně jsme dojili na naši spanici | konečně jsme dojeli na naši stanici |
| 7 | zívlena mi věit jaka to dobohu | zítra dám vědět jak to dopadlo |
| 8 | vývorho na opan je vysochá ronoz protikorozii | výhodou naopak je vysoká odolnost proti korozi |
| 9 | nasné a filbec nězděláme | jasné a všeobecně známé |
| 10 | můse jiško tí z klouds podporou | musíš to tisknout s podporou |
| 11 | nitro dlid cejin | nitroglycerin |

