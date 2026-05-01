"""
Convolutional Neural Networks (CNN) - Deep Dive
================================================

This module covers:
- CNN architecture
- Convolutional layers
- Pooling layers
- Building CNN for image classification
- Data augmentation
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense,
        Dropout, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("TensorFlow not available")

# ============================================================================
# 1. CNN CONCEPT
# ============================================================================

print("=" * 70)
print("1. CNN CONCEPT")
print("=" * 70)

print("""
Convolutional Neural Networks (CNN):

Designed for processing grid-like data (images)

Key Components:
1. Convolutional Layer: Extract features using filters
2. Pooling Layer: Reduce spatial dimensions
3. Fully Connected Layer: Make predictions

Why CNN for Images:
- Automatically learns features from raw pixels
- Translation invariant (finds features anywhere)
- Parameter efficient (vs fully connected)
""")

# ============================================================================
# 2. CONVOLUTIONAL LAYER
# ============================================================================

print("\n" + "=" * 70)
print("2. CONVOLUTIONAL LAYER")
print("=" * 70)

print("""
Convolutional Operation:
- Slide a filter (kernel) over the image
- At each position, compute dot product
- Result = feature map

Key Parameters:
- Filters: Number of feature maps to learn
- Kernel size: Size of the filter (3x3, 5x5)
- Stride: How far to move the filter
- Padding: Add border around image

Common Settings:
- Filters: 32, 64, 128, 256
- Kernel: 3x3 (most common)
- Padding: 'same' (keep size) or 'valid' (reduce)
""")

if CNN_AVAILABLE:
    # Visualize filters
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Simple edge detection filters
    filters = [
        # Vertical edge
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        # Horizontal edge
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        # Diagonal edges
        np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
        np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    ]

    for i, f in enumerate(filters):
        ax = axes[i // 4, i % 4]
        ax.imshow(f, cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')

    plt.suptitle('Example CNN Filters')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 3. POOLING LAYER
# ============================================================================

print("\n" + "=" * 70)
print("3. POOLING LAYER")
print("=" * 70)

print("""
Pooling: Downsample feature maps

Max Pooling:
- Take maximum value in each window
- Preserves strongest features
- Most common

Average Pooling:
- Take average value in each window
- Smoother representation

Common Settings:
- Pool size: 2x2
- Stride: 2 (no overlap)
""")

# ============================================================================
# 4. BUILDING A CNN
# ============================================================================

print("\n" + "=" * 70)
print("4. BUILDING A CNN")
print("=" * 70)

if CNN_AVAILABLE:
    # Simple CNN architecture
    cnn_model = Sequential([
        # First conv block
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        # Second conv block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third conv block
        Conv2D(64, (3, 3), activation='relu'),

        # Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    cnn_model.summary()

# ============================================================================
# 5. CNN FOR MNIST
# ============================================================================

print("\n" + "=" * 70)
print("5. CNN FOR MNIST")
print("=" * 70)

if CNN_AVAILABLE:
    # Load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape and normalize
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    # One-hot encode
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Build CNN
    mnist_cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    mnist_cnn.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = mnist_cnn.fit(
        X_train, y_train_cat,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = mnist_cnn.evaluate(X_test, y_test_cat)
    print(f"\nCNN Test Accuracy: {test_acc:.4f}")

# ============================================================================
# 6. DATA AUGMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("6. DATA AUGMENTATION")
print("=" * 70)

print("""
Data Augmentation: Artificially increase training data

Common Techniques:
- Rotation
- Flip (horizontal/vertical)
- Zoom
- Shift
- Shear

Why it works:
- Reduces overfitting
- Makes model more robust
- Simulates real-world variations
""")

if CNN_AVAILABLE:
    # Create data generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    # Visualize augmented images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Original image
    img = X_train[0]

    axes[0, 0].imshow(img.squeeze(), cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Augmented images
    for i in range(1, 10):
        augmented = datagen.random_transform(img)
        axes[i // 5, i % 5].imshow(augmented.squeeze(), cmap='gray')
        axes[i // 5, i % 5].set_title(f'Augmented {i}')
        axes[i // 5, i % 5].axis('off')

    plt.suptitle('Data Augmentation Examples')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. CNN ARCHITECTURE EVOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("7. CNN ARCHITECTURES")
print("=" * 70)

print("""
Classic CNN Architectures:

1. LeNet-5 (1998):
   - First CNN
   - Used for digit recognition
   - Simple architecture

2. AlexNet (2012):
   - Won ImageNet competition
   - 8 layers
   - ReLU activation

3. VGG (2014):
   - 16-19 layers
   - Small 3x3 filters
   - Very deep

4. ResNet (2015):
   - Skip connections
   - 152 layers
   - Solved vanishing gradient

Key Innovation: Skip Connections:
- Skip layers during forward pass
- Helps gradient flow
- Enables very deep networks
""")

# ============================================================================
# 8. TRANSFER LEARNING
# ============================================================================

print("\n" + "=" * 70)
print("8. TRANSFER LEARNING")
print("=" * 70)

print("""
Transfer Learning:
- Use pre-trained model as starting point
- Fine-tune on your data
- Great for small datasets

Popular Pre-trained Models:
- VGG16, VGG19
- ResNet50
- InceptionV3
- MobileNet

Two Approaches:
1. Feature Extraction: Freeze base, train new classifier
2. Fine-tuning: Unfreeze some layers, train end-to-end
""")

if CNN_AVAILABLE:
    # Load pre-trained VGG16
    from tensorflow.keras.applications import VGG16

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    vgg.summary()

    print("\nTransfer Learning Process:")
    print("1. Load pre-trained model")
    print("2. Freeze base layers")
    print("3. Add custom classifier")
    print("4. Train on your data")

# ============================================================================
# 9. PRACTICAL TIPS
# ============================================================================

print("\n" + "=" * 70)
print("9. PRACTICAL TIPS")
print("=" * 70)

print("""
CNN Best Practices:

1. Start Simple:
   - Basic CNN first
   - Add complexity if needed

2. Architecture Guidelines:
   - Increase filters as you go deeper
   - 32 → 64 → 128
   - Use 3x3 kernels

3. Regularization:
   - Dropout (0.3-0.5)
   - Data augmentation
   - Batch normalization

4. Training:
   - Use GPU if available
   - Start with good learning rate (0.001)
   - Monitor validation accuracy

5. Common Problems:
   - Overfitting → Add dropout/data augmentation
   - Underfitting → Increase model capacity
   - Slow training → Use GPU
""")

# ============================================================================
# 10. BUILD CUSTOM CNN
# ============================================================================

print("\n" + "=" * 70)
print("10. CUSTOM CNN EXAMPLE")
print("=" * 70)

if CNN_AVAILABLE:
    # CIFAR-10 dataset
    (X_cifar, y_cifar), (X_cifar_test, y_cifar_test) = keras.datasets.cifar10.load_data()

    print(f"CIFAR-10 shape: {X_cifar.shape}")
    print(f"Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck")

    # Normalize
    X_cifar = X_cifar.astype('float32') / 255.0
    X_cifar_test = X_cifar_test.astype('float32') / 255.0

    # Simple CNN for CIFAR-10
    cifar_cnn = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    cifar_cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train for 2 epochs (demo)
    history = cifar_cnn.fit(
        X_cifar[:5000], y_cifar[:5000],
        epochs=2,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    print("\nNote: Train for more epochs for better accuracy")

print("\n" + "=" * 70)
print("CNN SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. CNNs are designed for image data
2. Conv layers extract features
3. Pooling reduces dimensions
4. Use data augmentation to prevent overfitting
5. Transfer learning for small datasets
6. Popular architectures: LeNet, AlexNet, VGG, ResNet

When to use CNN:
- Image classification
- Object detection
- Image segmentation
- Any spatial data
""")
