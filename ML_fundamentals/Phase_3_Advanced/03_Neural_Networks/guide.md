# 🧠 Neural Networks — Deep Dive Guide

## 📋 Table of Contents

1. [Overview & Intuition](#overview--intuition)
2. [Learning Roadmap](#learning-roadmap)
3. [Perceptron & Neurons](#1-perceptron--neurons)
4. [Activation Functions](#2-activation-functions)
5. [Feedforward Networks (MLP)](#3-feedforward-networks-mlp)
6. [Backpropagation](#4-backpropagation)
7. [CNNs for Images](#5-convolutional-neural-networks-cnn)
8. [RNNs for Sequences](#6-recurrent-neural-networks-rnn)
9. [Practical Tips](#7-practical-tips)
10. [Key Takeaways](#key-takeaways)

---

## Overview & Intuition

**Core Idea**: Neural networks are composed of layers of interconnected "neurons" that learn to transform inputs into outputs through training.

### The Big Picture

```
Input Layer        Hidden Layers         Output Layer
(features)         (learned repr.)       (predictions)

  x₁  ──┐
         ├──→  [h₁] ──┐
  x₂  ──┤             ├──→  [h₃] ──┐
         ├──→  [h₂] ──┤            ├──→  ŷ
  x₃  ──┘             └──→  [h₄] ──┘

Each arrow has a WEIGHT (learned during training)
Each neuron has a BIAS + ACTIVATION FUNCTION
```

### Why Neural Networks?

| Traditional ML | Neural Networks |
|---|---|
| Manual feature engineering | Automatic feature learning |
| Fixed representations | Learned representations |
| Shallow models | Can be very deep |
| Good on tabular data | State-of-art on images, text, audio |

---

## Learning Roadmap

| Day | Topic | Study File | Time |
|-----|-------|-----------|------|
| 1-2 | Neurons, activations, forward pass | `01_neural_networks_basics.py` | 3-4h |
| 3-4 | Deep networks, regularization, optimizers | `02_deep_networks.py` | 3-4h |
| 5-6 | CNNs for image data | `03_cnn_images.py` | 3-4h |
| 7-8 | RNNs for sequential data | `04_rnn_sequences.py` | 3-4h |
| 9 | Exercises | `exercises.py` | 3-4h |

---

## 1. Perceptron & Neurons

### Single Neuron (Perceptron)

```
Inputs      Weights      Sum          Activation    Output
  x₁ ──── w₁ ──┐
                ├──→ Σ(wᵢxᵢ + b) ──→ f(z) ──→ ŷ
  x₂ ──── w₂ ──┤
                │
  bias ── b ────┘

  z = w₁x₁ + w₂x₂ + b        (linear combination)
  ŷ = f(z)                     (activation function)
```

### Learning Process

```
1. Forward Pass:    z = Wx + b → ŷ = f(z)
2. Compute Loss:    L = loss(y, ŷ)
3. Backward Pass:   ∂L/∂W, ∂L/∂b   (gradients)
4. Update Weights:  W = W - η × ∂L/∂W
                    b = b - η × ∂L/∂b
5. Repeat until convergence
```

---

## 2. Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary output, gates |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers (centered) |
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers (default!) |
| **Leaky ReLU** | max(0.01x, x) | (-∞, ∞) | When ReLU has dying neurons |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1) | Multi-class output |

### ReLU is King 👑

```
ReLU: f(x) = max(0, x)

Why ReLU?
  ✅ Fast computation (just a threshold)
  ✅ No vanishing gradient (for positive values)
  ✅ Sparse activation (efficiency)

  ⚠️ Dying ReLU problem: if neuron outputs 0, gradient = 0, never recovers
  → Solution: Leaky ReLU, ELU, or careful initialization
```

---

## 3. Feedforward Networks (MLP)

**Multi-Layer Perceptron** = multiple layers of neurons:

```
Input (4) → Hidden₁ (128, ReLU) → Hidden₂ (64, ReLU) → Output (3, Softmax)

Number of parameters:
  Layer 1: 4 × 128 + 128 = 640       (weights + biases)
  Layer 2: 128 × 64 + 64 = 8,256
  Layer 3: 64 × 3 + 3 = 195
  Total: 9,091 parameters
```

### When to Use MLP

```
✅ Tabular/structured data
✅ Classification and regression
✅ When you have enough data (>1000 samples)
✅ When feature interactions matter

❌ Not for images (use CNN)
❌ Not for sequences (use RNN/Transformer)
❌ Not for very small datasets (use traditional ML)
```

---

## 4. Backpropagation

### Chain Rule in Action

```
Forward:  x → z₁ = W₁x + b₁ → a₁ = ReLU(z₁) → z₂ = W₂a₁ + b₂ → ŷ = softmax(z₂)

Loss: L = CrossEntropy(y, ŷ)

Backward (chain rule):
  ∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂
  ∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂a₁ × ∂a₁/∂z₁ × ∂z₁/∂W₁

  Gradients flow BACKWARD through the network → "backpropagation"
```

---

## 5. Convolutional Neural Networks (CNN)

### Architecture

```
Input Image → [Conv → ReLU → Pool] × N → Flatten → Dense → Output

Example (image classification):
  Input: 28×28×1 (grayscale image)
  Conv1: 32 filters, 3×3 → 26×26×32
  Pool1: 2×2 max pool   → 13×13×32
  Conv2: 64 filters, 3×3 → 11×11×64
  Pool2: 2×2 max pool   → 5×5×64
  Flatten: 1600
  Dense: 128, ReLU
  Output: 10, Softmax (10 classes)
```

### Key CNN Concepts

| Component | What it does | Why |
|-----------|-------------|-----|
| **Convolution** | Slides filter over image, detects patterns | Local feature detection |
| **Pooling** | Downsamples feature maps | Reduces parameters, adds invariance |
| **Stride** | Step size of filter | Controls output size |
| **Padding** | Adds zeros around edges | Preserves spatial dimensions |
| **Filters** | Learned pattern detectors | Low-level: edges; High-level: objects |

---

## 6. Recurrent Neural Networks (RNN)

### Architecture

```
For sequence: [x₁, x₂, x₃, ..., xₜ]

  x₁ → [RNN] → h₁
         ↓
  x₂ → [RNN] → h₂     (same weights, shared across time steps)
         ↓
  x₃ → [RNN] → h₃
         ↓
  xₜ → [RNN] → hₜ → output

  hₜ = f(W_hh × hₜ₋₁ + W_xh × xₜ + b)
  Hidden state hₜ carries information from all previous steps
```

### RNN Variants

| Variant | Solves | Key Feature |
|---------|--------|-------------|
| **Vanilla RNN** | - | Simple, but vanishing gradient |
| **LSTM** | Vanishing gradient | Cell state + 3 gates (forget, input, output) |
| **GRU** | Vanishing gradient | Simpler than LSTM, 2 gates (reset, update) |
| **Bidirectional** | Context from both directions | Forward + backward passes |

---

## 7. Practical Tips

### Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| Overfitting | Dropout, early stopping, data augmentation |
| Vanishing gradient | ReLU, batch normalization, residual connections |
| Slow convergence | Adam optimizer, learning rate scheduling |
| Bad initialization | He/Xavier initialization |
| Unstable training | Gradient clipping, batch normalization |

### Optimizer Choice

```
SGD       → Simple, needs LR tuning, good generalization
Adam      → Good default, adaptive LR, fast convergence
RMSprop   → Good for RNNs
AdamW     → Adam + weight decay, state-of-the-art
```

### Learning Rate Strategy

```
Start: 0.001 (Adam) or 0.01 (SGD)
Schedule:
  • ReduceLROnPlateau: reduce when validation loss plateaus
  • CosineAnnealing: smooth decrease over training
  • WarmupCosine: warm up → cosine decay (transformers)
```

---

## Key Takeaways

1. **Start simple**: MLP → CNN/RNN → complex architectures
2. **ReLU** is the default activation for hidden layers
3. **Adam** optimizer is the default starting choice
4. **Batch Normalization** helps training stability
5. **Dropout** is the simplest regularization technique
6. **CNNs** for spatial data (images), **RNNs** for sequential data (text, time series)
7. **Monitor both train and val loss** to detect overfitting

### Architecture Decision

```
Tabular data       → MLP (or stick with XGBoost!)
Images             → CNN (ResNet, EfficientNet)
Text/Sequences     → RNN/LSTM/GRU (or Transformer)
Time Series        → RNN/LSTM or 1D-CNN
Audio              → 1D-CNN or spectrogram + 2D-CNN
```

---

## Study Files

| # | File | Description | Difficulty |
|---|------|-------------|------------|
| 1 | `01_neural_networks_basics.py` | Perceptron, MLP with sklearn & Keras | ⭐⭐ |
| 2 | `02_deep_networks.py` | Deep networks, dropout, batch norm, optimizers | ⭐⭐⭐ |
| 3 | `03_cnn_images.py` | CNN for image classification | ⭐⭐⭐ |
| 4 | `04_rnn_sequences.py` | RNN/LSTM for sequences | ⭐⭐⭐⭐ |
| 5 | `exercises.py` | 5 practice problems | ⭐⭐⭐ |
