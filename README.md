# Neural Machine Translation (NMT) Using Deep Learning

## Overview
This project develops a high-precision English-French translation model using a Seq2Seq Encoder-Decoder architecture with an Attention mechanism. The model is designed to enhance language translation accuracy significantly.

## Features
- Seq2Seq model with Attention for improved translation quality
- Dual LSTM layers in the encoder and a single LSTM layer in the decoder
- Utilization of NLTK for text preprocessing and GloVe embeddings for better input representation
- Trained on a diverse English-French dataset with 96% validation accuracy

## Dataset
The dataset used for training is sourced from [ManyThings.org](http://www.manythings.org/anki/fra-eng.zip), which contains parallel English-French sentence pairs.

## Model Architecture
- **Encoder:** Two stacked LSTM layers
- **Decoder:** Single LSTM layer with an Attention mechanism
- **Embedding Layer:** Uses GloVe embeddings for better word representation
- **Loss Function:** Categorical cross-entropy
- **Optimizer:** Adam optimizer

## Training the Model
- The model trains for 100 epochs with a batch size of 64.
- The training and validation split is 80-20.

## Inference and Translation
Example Output:
```
Input: "Hello, how are you?"
Output: "Bonjour, comment ça va?"
```

## Results
- The model achieves **96% validation accuracy** on the dataset.
- Example Translation:
  - **Input:** "Run."
  - **Translated Output:** "Fuyons!"

## Model Checkpoints
Trained model weights are saved in the `/Models/` directory:
```
Models/
    fren_to_eng.keras
```

## Future Improvements
- Experimenting with Transformer-based models for further performance gains
- Training on larger datasets for improved generalization
- Implementing Beam Search for enhanced translation fluency

## Contact
For any queries, feel free to reach out:
- **Name:** Manan Parakh
- **Email:** [mananparakh500@example.com](mailto:mananparakh500@example.com)
- **GitHub:** [Manan-Parakh](https://github.com/Manan-Parakh)

