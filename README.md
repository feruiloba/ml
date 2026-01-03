# Machine Learning Coursework Repository

This repository contains assignments from three graduate-level Machine Learning courses.

## Setup

Install Python using [PyEnv](https://github.com/pyenv/pyenv):
```bash
pyenv install 3.12.4  # for deep_learning/hw4, gen_ai
pyenv install 3.9.12  # for intro_to_ml
pyenv global 3.12.4
```

Each homework has its own virtual environment:
```bash
# Activate environment
source <hw_folder>/.venv/bin/activate
# or for older assignments:
source <hw_folder>/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Introduction to Machine Learning (intro_to_ml/)

Classical ML algorithms implemented from scratch.

### HW1 - Majority Vote Classifier
Simple baseline classifier using majority voting.
```bash
cd intro_to_ml/hw1
python majority_vote.py heart_train.tsv heart_test.tsv heart_train_labels.txt heart_test_labels.txt heart_metrics.txt
```

### HW2 - Decision Tree (ID3)
ID3 decision tree algorithm with configurable max depth.
```bash
cd intro_to_ml/hw2
python decision_tree.py heart_train.tsv heart_test.tsv 2 heart_2_train.txt heart_2_test.txt heart_2_metrics.txt heart_2_print.txt
```

### HW4 - Logistic Regression
Feature extraction with GloVe embeddings and logistic regression classifier.
```bash
cd intro_to_ml/hw4
python feature.py smalldata/train_small.tsv smalldata/val_small.tsv smalldata/test_small.tsv glove_embeddings.txt formatted_train_small.tsv formatted_val_small.tsv formatted_test_small.tsv
python lr.py formatted_train_small.tsv formatted_val_small.tsv formatted_test_small.tsv train_labels_out_small.txt test_labels_out_small.txt metrics_small.txt 500 0.1
```

### HW5 - Neural Network
Single hidden layer neural network for classification.
```bash
cd intro_to_ml/hw5
python neuralnet.py small_train.csv small_validation.csv small_train_out.labels small_validation_out.labels small_metrics_out.txt 2 4 2 0.1
```

### HW7 - RNN with Self-Attention
RNN language model with self-attention mechanism.
```bash
cd intro_to_ml/hw7
python rnn.py --train_data data/tiny_train_stories.json --val_data data/tiny_valid_stories.json --metrics_out metrics.txt --train_losses_out train_losses.txt --val_losses_out val_losses.txt --embed_dim 64 --hidden_dim 128 --dk 32 --dv 32 --num_sequences 128 --batch_size 1
```

### HW8 - Q-Learning
Reinforcement learning with Q-learning and tile coding.
```bash
cd intro_to_ml/hw8
python q_learning.py mc tile mc_tile_100_200_0.0_0.99_0.005_0_0_0_weights.txt mc_tile_100_200_0.0_0.99_0.005_0_0_0_returns.txt 100 200 0.0 0.99 0.005 1 200 3
```

---

## Introduction to Deep Learning (deep_learning/)

Building neural network components from scratch in `mytorch/`, progressively implementing more complex architectures.

### HW1 - MLP Fundamentals
Implements linear layers, activations (ReLU, Sigmoid), batch normalization, loss functions (MSE, CrossEntropy), and SGD optimizer.
```bash
cd deep_learning/hw1
source bin/activate
python autograder/hw1p1_autograder.py
```

### HW2 - Convolutional Neural Networks
Implements Conv1d, Conv2d, ConvTranspose, pooling, and resampling layers.
```bash
cd deep_learning/hw2
python autograder/runner.py
```

### HW3P1 - Recurrent Neural Networks
Implements RNN cells, GRU cells, CTC loss, and CTC decoding (greedy + beam search).
```bash
cd deep_learning/hw3p1
python autograder/runner.py
```

### HW4P1 - Transformer (Decoder-Only)
Causal language modeling with decoder-only Transformer. Implements multi-head attention, positional encoding, and decoder layers.
```bash
cd deep_learning/hw4p1
source .venv/bin/activate
python -m pytest tests/test_mytorch.py
python -m pytest tests/test_transformer_decoder_only.py
```
Primary development in `HW4P1_Student_Notebook.ipynb`.

### HW4P2 - Transformer (Encoder-Decoder)
End-to-end speech recognition with encoder-decoder Transformer.
```bash
cd deep_learning/hw4p1
python -m pytest tests/test_transformer_encoder_decoder.py
python -m pytest tests/test_decoding.py
```
Primary development in `HW4P2_Student_Starter_Notebook.ipynb`.

---

## Generative AI (gen_ai/)

Implementing and training generative models.

### HW0 - Classifiers Baseline
Image and text classification warmup.
```bash
cd gen_ai/hw0
source bin/activate
python img_classifier.py
python txt_classifier.py
```

### HW1 - GPT (Character-Level)
Character-level GPT implementation using minGPT framework. Implements multi-head attention with optional RoPE.
```bash
cd gen_ai/hw1
source lib/bin/activate
python chargpt.py
python test_model.py  # run tests
```

### HW2 - Diffusion Models (DDPM)
Denoising Diffusion Probabilistic Models with U-Net architecture.
```bash
cd gen_ai/hw2
source .venv/bin/activate
python main.py --time_steps 50 --train_steps 10000 --data_class cat --batch_size 32
python test_diffusion.py  # run tests
```

### HW3 - LoRA Fine-tuning
Low-Rank Adaptation for efficient GPT-2 fine-tuning.
```bash
cd gen_ai/hw3
source .venv/bin/activate
python train.py --out_dir lora-gpt --init_from gpt2 --lora_rank 128 --lora_alpha 512
python generate.py  # generate samples
python test_lora.py  # run tests
```

### HW4 - DiT and Q-Former
Diffusion Transformer (DiT) and Q-Former for text-conditioned image generation.
```bash
cd gen_ai/hw4
source .venv/bin/activate
python train_qformer.py --epochs 10 --batch_size 32 --lr 1e-4
python eval_qformer.py  # evaluate model
python test_all.py  # run tests
```
