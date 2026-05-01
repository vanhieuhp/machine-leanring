"""
Computer Vision - Part 1: CNN Fundamentals
=========================================

This module covers:
- Image representation
- Convolutional operations
- Building CNN from scratch
- Training and evaluation
- PyTorch implementation

Based on: CIFAR-10 Image Classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. IMAGE REPRESENTATION
# ============================================================================

print("=" * 70)
print("1. IMAGE REPRESENTATION")
print("=" * 70)

# Create sample image data (CIFAR-10 like)
np.random.seed(42)

# Simulate 32x32 RGB images
n_samples = 100
height, width, channels = 32, 32, 3

# Generate random images
images = np.random.randint(0, 256, (n_samples, height, width, channels), dtype=np.uint8)

# Labels (0-9 for 10 classes)
labels = np.random.randint(0, 10, n_samples)

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Image shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"\nClass distribution:")
for i, name in enumerate(class_names):
    count = np.sum(labels == i)
    print(f"  {i}. {name}: {count} samples")

# -------------------------------------------------------------------------
# 1.1 Image Normalization
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.1 Image Normalization")
print("-" * 50)

# Normalize to [0, 1]
images_normalized = images / 255.0

print(f"Original range: [{images.min()}, {images.max()}]")
print(f"Normalized range: [{images_normalized.min():.2f}, {images_normalized.max():.2f}]")

# Standardize (ImageNet mean/std)
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

# For now, just normalize to [0, 1]
print(f"\nImageNet mean: {imagenet_mean}")
print(f"ImageNet std: {imagenet_std}")

# ============================================================================
# 2. CONVOLUTIONAL OPERATION
# ============================================================================

print("\n" + "=" * 70)
print("2. CONVOLUTIONAL OPERATION")
print("=" * 70)

# Simple 2D convolution
def conv2d(image, kernel, stride=1, padding=0):
    """
    Perform 2D convolution.

    Parameters:
    -----------
    image : 2D array
        Input image
    kernel : 2D array
        Convolution kernel/filter
    stride : int
        Stride
    padding : int
        Padding

    Returns:
    --------
    output : 2D array
        Convolved output
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    # Calculate output size
    h, w = image.shape
    k_h, k_w = kernel.shape
    out_h = (h - k_h) // stride + 1
    out_w = (w - k_w) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(0, h - k_h + 1, stride):
        for j in range(0, w - k_w + 1, stride):
            region = image[i:i+k_h, j:j+k_w]
            output[i//stride, j//stride] = np.sum(region * kernel)

    return output

# Create sample image
sample_image = np.random.randn(10, 10)

# Create edge detection kernel (Sobel)
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Create blur kernel
blur_kernel = np.ones((3, 3)) / 9

# Apply convolutions
edge_result = conv2d(sample_image, edge_kernel)
blur_result = conv2d(sample_image, blur_kernel)

print(f"Original shape: {sample_image.shape}")
print(f"Edge detection output: {edge_result.shape}")
print(f"Blur output: {blur_result.shape}")

# ============================================================================
# 3. PYTORCH CNN
# ============================================================================

print("\n" + "=" * 70)
print("3. PYTORCH CNN IMPLEMENTATION")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    print(f"PyTorch version: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 3.1 Define CNN Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.1 Defining CNN Architecture")
    print("-" * 50)

    class SimpleCNN(nn.Module):
        """Simple CNN for image classification."""

        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()

            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)

            # Pooling
            self.pool = nn.MaxPool2d(2, 2)

            # Dropout
            self.dropout = nn.Dropout(0.5)

            # Fully connected layers
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            # Conv block 1
            x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 -> 16

            # Conv block 2
            x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16 -> 8

            # Conv block 3
            x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8 -> 4

            # Flatten
            x = x.view(x.size(0), -1)

            # Fully connected
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)

            return x

    # Instantiate model
    model = SimpleCNN(num_classes=10)
    print("Model architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # -------------------------------------------------------------------------
    # 3.2 Prepare Data
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.2 Preparing Data")
    print("-" * 50)

    # Create synthetic data (in practice, use CIFAR-10 dataset)
    n_train = 1000
    n_test = 200

    # Random training data
    X_train = np.random.rand(n_train, 3, 32, 32).astype(np.float32)
    y_train = np.random.randint(0, 10, n_train)

    X_test = np.random.rand(n_test, 3, 32, 32).astype(np.float32)
    y_test = np.random.randint(0, 10, n_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # -------------------------------------------------------------------------
    # 3.3 Training
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.3 Training CNN")
    print("-" * 50)

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    model.train()

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        # Print progress
        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    # -------------------------------------------------------------------------
    # 3.4 Evaluation
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.4 Evaluating CNN")
    print("-" * 50)

    model.eval()
    with torch.no_grad():
        # Test predictions
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        test_accuracy = (predicted == y_test_tensor).float().mean()
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Per-class accuracy
        print("\nPer-class accuracy:")
        for i in range(10):
            mask = y_test_tensor == i
            if mask.sum() > 0:
                class_acc = (predicted[mask] == y_test_tensor[mask]).float().mean()
                print(f"  Class {i}: {class_acc:.4f}")

except ImportError as e:
    print(f"PyTorch not available: {e}")
    print("Install with: pip install torch")

# ============================================================================
# 4. KERAS CNN
# ============================================================================

print("\n" + "=" * 70)
print("4. KERAS/TENSORFLOW CNN IMPLEMENTATION")
print("=" * 70)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping

    print(f"TensorFlow version: {tf.__version__}")

    # -------------------------------------------------------------------------
    # 4.1 Define Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("4.1 Defining Keras CNN Model")
    print("-" * 50)

    keras_model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Compile
    keras_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model summary:")
    keras_model.summary()

    # -------------------------------------------------------------------------
    # 4.2 Train Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("4.2 Training Keras CNN")
    print("-" * 50)

    # Create dummy data
    X_train_keras = np.random.rand(1000, 32, 32, 3).astype(np.float32)
    y_train_keras = np.random.randint(0, 10, 1000)
    X_test_keras = np.random.rand(200, 32, 32, 3).astype(np.float32)
    y_test_keras = np.random.randint(0, 10, 200)

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train
    history = keras_model.fit(
        X_train_keras, y_train_keras,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # -------------------------------------------------------------------------
    # 4.3 Evaluate
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("4.3 Evaluating Keras CNN")
    print("-" * 50)

    # Evaluate
    test_loss, test_acc = keras_model.evaluate(X_test_keras, y_test_keras, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predictions
    predictions = keras_model.predict(X_test_keras, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    print(f"\nPrediction distribution:")
    for i in range(10):
        count = np.sum(pred_classes == i)
        print(f"  Class {i}: {count}")

except ImportError as e:
    print(f"TensorFlow not available: {e}")
    print("Install with: pip install tensorflow")

# ============================================================================
# 5. DATA AUGMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("5. DATA AUGMENTATION")
print("=" * 70)

try:
    from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("\n" + "-" * 50)
    print("5.1 Keras Data Augmentation")
    print("-" * 50)

    # Create data generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    print("Augmentation techniques:")
    print("  - Rotation: 20 degrees")
    print("  - Width shift: 20%")
    print("  - Height shift: 20%")
    print("  - Horizontal flip")
    print("  - Zoom: 20%")

    # Using Keras preprocessing layers
    print("\n" + "-" * 50)
    print("5.2 Keras Preprocessing Layers")
    print("-" * 50)

    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

    print("Preprocessing layers:")
    data_augmentation.summary()

except ImportError:
    print("TensorFlow required for this section")

# ============================================================================
# 6. COMPLETE PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("6. COMPLETE IMAGE CLASSIFICATION PIPELINE")
print("=" * 70)

def cnn_pipeline(X_train, y_train, X_test, y_test, epochs=10):
    """
    Complete CNN pipeline for image classification.

    Parameters:
    -----------
    X_train, X_test : arrays
        Image data
    y_train, y_test : arrays
        Labels
    epochs : int
        Training epochs

    Returns:
    --------
    model, history
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test Accuracy: {test_acc:.4f}")

    return model, history

# Test pipeline
print("Testing complete pipeline...")
X_train_dummy = np.random.rand(500, 32, 32, 3).astype(np.float32)
y_train_dummy = np.random.randint(0, 10, 500)
X_test_dummy = np.random.rand(100, 32, 32, 3).astype(np.float32)
y_test_dummy = np.random.randint(0, 10, 100)

try:
    model, history = cnn_pipeline(X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy, epochs=5)
except:
    print("Pipeline test skipped (TensorFlow not available)")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Image Representation
   - Images as arrays (height × width × channels)
   - Normalize to [0, 1] or standardize
   - RGB = 3 channels, Grayscale = 1 channel

2. Convolutional Operation
   - Kernel slides over image
   - Element-wise multiplication and sum
   - Extracts features (edges, textures, etc.)

3. CNN Architecture
   - Conv layers: Feature extraction
   - Pooling: Spatial reduction
   - BatchNorm: Stabilize training
   - Dropout: Regularization
   - FC layers: Classification

4. Training Tips
   - Use data augmentation
   - Batch normalization helps
   - Start with simple models
   - Monitor validation loss

5. Data Augmentation
   - Increase effective dataset size
   - Reduces overfitting
   - Common: flip, rotate, zoom, crop

Next: Transfer Learning (02_transfer_learning.py)
""")
