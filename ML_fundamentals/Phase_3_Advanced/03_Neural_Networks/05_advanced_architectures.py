"""
Advanced Neural Network Architectures
=====================================

This module covers:
- Modern architectures
- ResNet skip connections
- Inception modules
- Attention mechanism
- Transformers basics
- Transfer learning
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

# ============================================================================
# 1. RESNET - SKIP CONNECTIONS
# ============================================================================

print("=" * 70)
print("1. RESNET - SKIP CONNECTIONS")
print("=" * 70)

print("""
ResNet: Residual Network

Key Innovation: Skip Connections (Residual Connections)

Problem:
- Very deep networks had worse performance
- Degradation problem (not overfitting)

Solution:
- Add skip connections
- Learn residual: F(x) = H(x) - x
- Makes training deeper networks possible

Why it works:
- Gradient flows directly through skip connection
- Easier to learn identity mapping
- Deeper networks can be trained
""")

if ADVANCED_AVAILABLE:
    from tensorflow.keras.layers import Add, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D

    # Simple ResNet block
    def res_block(x, filters, kernel_size=3):
        shortcut = x

        # Main path
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)

        # Add shortcut
        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        return x

    # Build simple ResNet
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, 3, padding='same', activation='relu')(inputs)

    # Add ResNet blocks
    x = res_block(x, 32)
    x = res_block(x, 32)

    x = MaxPooling2D()(x)

    x = res_block(x, 64)
    x = res_block(x, 64)

    model = keras.Model(inputs=inputs, outputs=x)
    model.summary()

# ============================================================================
# 2. INCEPTION MODULE
# ============================================================================

print("\n" + "=" * 70)
print("2. INCEPTION MODULE")
print("=" * 70)

print("""
Inception Network (Google):

Key Idea: Don't choose filter size, use all!

Inception Module:
- 1x1 convolutions (reduce dimensions)
- 3x3 convolutions
- 5x5 convolutions
- Max pooling
- Concatenate all outputs

Benefits:
- Capture different scales
- More efficient than large filters
- GoogleNet (Inception v1) won 2014 ImageNet
""")

if ADVANCED_AVAILABLE:
    from tensorflow.keras.layers import Concatenate

    def inception_module(x, filters):
        # 1x1 conv
        conv1 = Conv2D(filters[0], 1, activation='relu')(x)

        # 1x1 then 3x3
        conv3 = Conv2D(filters[1], 1, activation='relu')(x)
        conv3 = Conv2D(filters[2], 3, padding='same', activation='relu')(conv3)

        # 1x1 then 5x5
        conv5 = Conv2D(filters[3], 1, activation='relu')(x)
        conv5 = Conv2D(filters[4], 5, padding='same', activation='relu')(conv5)

        # MaxPool then 1x1
        pool = MaxPooling2D(3, strides=1, padding='same')(x)
        pool = Conv2D(filters[5], 1, activation='relu')(pool)

        # Concatenate
        return Concatenate()([conv1, conv3, conv5, pool])

    print("Inception module applies multiple filter sizes in parallel")

# ============================================================================
# 3. ATTENTION MECHANISM
# ============================================================================

print("\n" + "=" * 70)
print("3. ATTENTION MECHANISM")
print("=" * 70)

print("""
Attention: Focus on what matters

What is Attention:
- Neural network component
- Assigns importance weights to inputs
- "Attend to" relevant parts

Types of Attention:
1. Additive (Bahdanau)
2. Multiplicative (Luong)
3. Scaled Dot-Product (Transformer)

Key Benefits:
- Handles long-range dependencies
- Interpretable (see what model attends to)
- Parallelizable (unlike RNN)
""")

# Visualize attention
if ADVANCED_AVAILABLE:
    # Simulated attention weights
    seq_length = 10
    attention_weights = np.random.rand(seq_length)
    attention_weights = attention_weights / attention_weights.sum()

    plt.figure(figsize=(10, 4))
    plt.bar(range(seq_length), attention_weights, color='steelblue')
    plt.xlabel('Position')
    plt.ylabel('Attention Weight')
    plt.title('Attention Weights Example')
    plt.xticks(range(seq_length))
    plt.show()

# ============================================================================
# 4. TRANSFORMERS
# ============================================================================

print("\n" + "=" * 70)
print("4. TRANSFORMERS")
print("=" * 70)

print("""
Transformer: Attention is All You Need (2017)

Architecture:
- Encoder: Process input
- Decoder: Generate output
- Self-Attention: Relate different positions

Key Components:
1. Multi-Head Attention
2. Position Encoding
3. Feed-Forward Networks
4. Layer Normalization

Why Transformers:
- Parallel processing (faster than RNN)
- Long-range dependencies
- State-of-the-art in NLP
- Now in vision (ViT)

BERT: Bidirectional Encoder
GPT: Generative Pre-training
""")

if ADVANCED_AVAILABLE:
    # Show transformer architecture conceptually
    print("""
    Transformer Architecture:

    Input → [Positional Encoding] → Encoder × N
                                    ↓
    Output ← [Linear + Softmax] ← Decoder × N
                                      ↑
    Output → [Positional Encoding] → Decoder

    Encoder: Self-Attention + Feed-Forward
    Decoder: Self-Attention + Encoder-Decoder Attention + Feed-Forward
    """)

# ============================================================================
# 5. TRANSFER LEARNING
# ============================================================================

print("\n" + "=" * 70)
print("5. TRANSFER LEARNING")
print("=" * 70)

print("""
Transfer Learning:
- Use pre-trained model weights
- Adapt to new task

Approaches:
1. Feature Extraction:
   - Freeze pre-trained layers
   - Train only new classifier

2. Fine-tuning:
   - Unfreeze some layers
   - Train end-to-end

Popular Pre-trained Models:
- ImageNet models: VGG, ResNet, EfficientNet
- NLP models: BERT, GPT, RoBERTa
""")

if ADVANCED_AVAILABLE:
    from tensorflow.keras.applications import ResNet50

    # Load pre-trained ResNet
    print("Loading pre-trained ResNet50...")
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print(f"ResNet50 layers: {len(resnet.layers)}")

# ============================================================================
# 6. FINE-TUNING
# ============================================================================

print("\n" + "=" * 70)
print("6. FINE-TUNING STRATEGY")
print("=" * 70)

print("""
Fine-tuning Steps:

1. Load pre-trained model
2. Freeze base layers
3. Train new classifier head
4. Optionally unfreeze some base layers
5. Train with low learning rate

Learning Rate:
- New layers: High (0.001)
- Fine-tuning: Low (0.0001)
- Use learning rate scheduler
""")

if ADVANCED_AVAILABLE:
    # Show fine-tuning approach
    print("""
    Fine-tuning Example:

    # 1. Load model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # 2. Freeze base
    for layer in base_model.layers:
        layer.trainable = False

    # 3. Add classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    # 4. Train
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(...)

    # 5. Optionally unfreeze
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    """)

# ============================================================================
# 7. EFFICIENT NETWORKS
# ============================================================================

print("\n" + "=" * 70)
print("7. EFFICIENT NETWORKS")
print("=" * 70)

print("""
Efficient Networks:
- Designed for efficiency
- Good accuracy with fewer parameters

EfficientNet (2019):
- Compound scaling
- Depth, width, resolution
- State-of-the-art efficiency

MobileNet:
- Designed for mobile/edge
- Depthwise separable convolutions
- Very fast inference
""")

# ============================================================================
# 8. GENERATIVE MODELS
# ============================================================================

print("\n" + "=" * 70)
print("8. GENERATIVE MODELS")
print("=" * 70)

print("""
Generative Models:

1. VAE (Variational Autoencoder):
   - Encoder → Latent space → Decoder
   - Learn to generate new samples

2. GAN (Generative Adversarial Network):
   - Generator: Create fake samples
   - Discriminator: Distinguish real/fake
   - Train together (adversarial)

3. Diffusion Models:
   - Slowly add noise
   - Learn to denoise
   - DALL-E, Stable Diffusion
""")

# ============================================================================
# 9. PRACTICAL ARCHITECTURE SELECTION
# ============================================================================

print("\n" + "=" * 70)
print("9. ARCHITECTURE SELECTION GUIDE")
print("=" * 70)

print("""
Quick Selection Guide:

Computer Vision:
- Classification: ResNet, EfficientNet
- Detection: YOLO, Faster R-CNN
- Segmentation: U-Net
- Mobile: MobileNet

NLP:
- Classification: BERT, RoBERTa
- Generation: GPT, T5
- Translation: Transformer

Time Series:
- LSTM/GRU
- Temporal Convolutional Network
- Transformer (for long sequences)
""")

# ============================================================================
# 10. MODEL COMPRESSION
# ============================================================================

print("\n" + "=" * 70)
print("10. MODEL COMPRESSION")
print("=" * 70)

print("""
Model Compression Techniques:

1. Pruning:
   - Remove unnecessary weights
   - Keep important connections

2. Quantization:
   - Use lower precision (int8 vs float32)
   - Reduce model size

3. Knowledge Distillation:
   - Train small model from large one
   - Transfer knowledge

4. Architecture:
   - Use efficient architectures
   - MobileNet, EfficientNet
""")

print("\n" + "=" * 70)
print("ADVANCED ARCHITECTURES SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. Skip connections enable very deep networks
2. Inception captures multiple scales
3. Attention is key to modern NLP
4. Transformers revolutionized NLP
5. Transfer learning for small datasets
6. Choose architecture based on task

State-of-the-art:
- Vision: Vision Transformers (ViT), EfficientNet
- NLP: Transformers (BERT, GPT)
- Generative: Diffusion models
""")
