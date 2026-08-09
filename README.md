# Deep Learning

This series of tasks is inspired by the [NPFL133 – Deep Learning](https://ufal.mff.cuni.cz/courses/npfl138/2526-summer) course and its associated assignments. The primary objective is to gain practical experience with the implementation of deep neural network architectures across a variety of machine learning tasks.

## Topics Covered

##### 1. Classification and Semantic Segmentation

- backbone model combined with classification and segmentation heads.
- Implementation of a Feature Pyramid Network (FPN) approach using transposed convolutions for segmentation.

##### 2. Object Detection

- backbone model with classification and regression heads for predicting object classes and bounding boxes.
- Implementation of an R-CNN-based approach for processing bounding box proposals.

##### 3. Recurrent Neural Networks for Part-of-Speech Tagging

- Use of character-level embeddings (CLE) and word embeddings as input representations.
- Processing of the resulting representations using bidirectional RNNs for part-of-speech tag prediction.

##### 4. Attention Mechanism for Lemmatization

- Implementation of an encoder-decoder architecture for sequence-to-sequence lemmatization.
- Integration of Bahdanau (additive) attention into the encoder-decoder architecture.

##### 5. Speech Recognition

- Processing of speech represented in MFCC format using an encoder based on stacked bidirectional LSTMs.
- Use of a classification head for output prediction.
- Training with Connectionist Temporal Classification (CTC) loss.

##### 6. Transformers for Tagging and Question Answering

- Implementation of transformer residual blocks and self-attention mechanisms.
- Application of the transformer architecture to Czech language tagging tasks.
- Use of a RoBERTa-based backbone for answer prediction from a given textual context in question-answering tasks.

##### 7. Generative Models

- Implementation of a Variational Autoencoder (VAE).
- Implementation of a Deep Convolutional Generative Adversarial Network (DCGAN).
- Application and evaluation of both models on the MNIST dataset.

##### 8. Flow Matching

- Implementation of a flow-matching model using a U-Net architecture for vector-field prediction based on an optimal transport formulation.
- Generation of images by numerically integrating the predicted vector field using the Euler method.