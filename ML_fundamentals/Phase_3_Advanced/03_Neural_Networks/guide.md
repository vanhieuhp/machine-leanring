# Neural Networks Guide

## What are Neural Networks?

Neural networks are computational models inspired by biological neurons. They learn complex patterns through layers of interconnected nodes.

## Basic Architecture

### 1. Input Layer
- Receives raw data
- One node per feature

### 2. Hidden Layers
- Process information
- Learn representations
- Multiple layers = deep learning

### 3. Output Layer
- Makes predictions
- One node per class (classification)
- One node for regression

## Key Components

### 1. Neurons
- Receive inputs
- Apply weights
- Add bias
- Apply activation function

### 2. Weights and Biases
- Learned parameters
- Updated during training
- Determine model behavior

### 3. Activation Functions
- Introduce non-linearity
- Common: ReLU, Sigmoid, Tanh

### 4. Loss Function
- Measures prediction error
- Minimized during training
- Examples: MSE, Cross-entropy

## Training Process

1. **Forward Pass**: Compute predictions
2. **Calculate Loss**: Measure error
3. **Backward Pass**: Compute gradients
4. **Update Weights**: Gradient descent
5. **Repeat**: Until convergence

## Architectures

### Feedforward Networks
- Simple, fully connected
- Good for tabular data

### Convolutional Networks (CNN)
- For images
- Learns spatial patterns
- Reduces parameters

### Recurrent Networks (RNN)
- For sequences
- Remembers previous inputs
- Good for time series, text

## Advantages

- Learn complex patterns
- End-to-end learning
- State-of-the-art performance
- Flexible architecture

## Disadvantages

- Requires lots of data
- Slow training
- Hard to interpret
- Hyperparameter tuning

## When to Use

- Large datasets
- Complex patterns
- Images, text, sequences
- When accuracy is critical

## Study Files

1. `01_neural_networks_basics.py` - Basic networks
2. `02_deep_networks.py` - Deep learning
3. `03_cnn_images.py` - Convolutional networks
4. `04_rnn_sequences.py` - Recurrent networks
5. `exercises.py` - Practice problems
