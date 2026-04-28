"""
Keras Basics - Deep Learning with TensorFlow/Keras
===================================================

This module covers:
- Building neural networks with Keras
- Sequential API
- Dense layers
- Training and evaluation
- Callbacks
- Model saving/loading
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Try to import Keras/TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.utils import to_categorical
    KERAS_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not installed. Install with: pip install tensorflow")

# ============================================================================
# 1. KERAS OVERVIEW
# ============================================================================

print("\n" + "=" * 70)
print("1. KERAS OVERVIEW")
print("=" * 70)

print("""
Keras: High-level neural networks API

Key Features:
- Simple and easy to use
- Runs on TensorFlow (or Theano, CNTK)
- Fast prototyping
- Built-in tools for training, evaluation, saving

Two ways to build models:
1. Sequential API: Simple, layer-by-layer
2. Functional API: More flexible, for complex architectures
""")

if not KERAS_AVAILABLE:
    print("\nPlease install TensorFlow to continue:")
    print("pip install tensorflow")

# ============================================================================
# 2. BASIC NEURAL NETWORK WITH KERAS
# ============================================================================

print("\n" + "=" * 70)
print("2. BASIC NEURAL NETWORK")
print("=" * 70)

if KERAS_AVAILABLE:
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=3, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Model summary
    model.summary()

# ============================================================================
# 3. TRAINING THE MODEL
# ============================================================================

print("\n" + "=" * 70)
print("3. TRAINING THE MODEL")
print("=" * 70)

if KERAS_AVAILABLE:
    # Train
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

# ============================================================================
# 4. VISUALIZING TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("4. VISUALIZING TRAINING")
print("=" * 70)

if KERAS_AVAILABLE:
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. CALLBACKS
# ============================================================================

print("\n" + "=" * 70)
print("5. CALLBACKS")
print("=" * 70)

print("""
Callbacks: Functions called at different stages during training

Common Callbacks:
1. EarlyStopping: Stop when validation doesn't improve
2. ModelCheckpoint: Save best model
3. ReduceLROnPlateau: Reduce learning rate when stuck
4. TensorBoard: Visualize training
""")

if KERAS_AVAILABLE:
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.001
    )

    # Rebuild and train with callbacks
    model2 = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model2.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history2 = model2.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    print(f"Training stopped at epoch: {len(history2.history['loss'])}")
    print(f"Best val_loss: {min(history2.history['val_loss']):.4f}")

# ============================================================================
# 6. MNIST EXAMPLE
# ============================================================================

print("\n" + "=" * 70)
print("6. MNIST EXAMPLE")
print("=" * 70)

if KERAS_AVAILABLE:
    # Load MNIST
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()

    print(f"Training data shape: {X_train_mnist.shape}")
    print(f"Test data shape: {X_test_mnist.shape}")

    # Normalize
    X_train_mnist = X_train_mnist / 255.0
    X_test_mnist = X_test_mnist / 255.0

    # Build model
    mnist_model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    mnist_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history_mnist = mnist_model.fit(
        X_train_mnist, y_train_mnist,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = mnist_model.evaluate(X_test_mnist, y_test_mnist)
    print(f"\nMNIST Test Accuracy: {test_acc:.4f}")

    # Visualize predictions
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test_mnist[i], cmap='gray')
        pred = np.argmax(mnist_model.predict(X_test_mnist[i:i+1], verbose=0))
        plt.title(f'Pred: {pred}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. MULTI-CLASS CLASSIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("7. MULTI-CLASS CLASSIFICATION")
print("=" * 70)

if KERAS_AVAILABLE:
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # One-hot encode
    y_cat = to_categorical(y, 3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    iris_model = Sequential([
        Dense(16, activation='relu', input_shape=(4,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    iris_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    iris_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate
    predictions = iris_model.predict(X_test_scaled)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(f"Iris Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=iris.target_names))

# ============================================================================
# 8. DROPOUT AND REGULARIZATION
# ============================================================================

print("\n" + "=" * 70)
print("8. DROPOUT AND REGULARIZATION")
print("=" * 70)

print("""
Dropout: Randomly set some activations to 0 during training
- Prevents overfitting
- Forces network to learn redundant representations
- Typical rate: 0.2 - 0.5

Other regularization techniques:
- L1/L2 regularization on weights
- Early stopping
- Data augmentation
""")

if KERAS_AVAILABLE:
    # Model with dropout
    model_dropout = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dropout(0.3),  # 30% dropout
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model_dropout.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Model with Dropout:")
    model_dropout.summary()

# ============================================================================
# 9. SAVING AND LOADING MODELS
# ============================================================================

print("\n" + "=" * 70)
print("9. SAVING AND LOADING MODELS")
print("=" * 70)

print("""
Save/Load Options:
1. Save entire model (including architecture, weights, optimizer):
   model.save('model.h5')
   model = keras.models.load_model('model.h5')

2. Save only weights:
   model.save_weights('weights.h5')

3. Save only architecture:
   model.to_json()
""")

if KERAS_AVAILABLE:
    # Save model
    # model.save('my_model.h5')

    # Load model
    # loaded_model = keras.models.load_model('my_model.h5')

    print("Model saving/loading code:")
    print("  # Save")
    print("  model.save('my_model.h5')")
    print("  ")
    print("  # Load")
    print("  model = keras.models.load_model('my_model.h5')")

# ============================================================================
# 10. OPTIMIZERS
# ============================================================================

print("\n" + "=" * 70)
print("10. OPTIMIZERS")
print("=" * 70)

print("""
Common Optimizers:

1. SGD (Stochastic Gradient Descent):
   - Classic optimizer
   - Can be slow
   - Needs learning rate scheduling

2. Adam (Adaptive Moment Estimation):
   - Most popular
   - Adaptive learning rates
   - Works well by default
   - Good for most cases

3. RMSprop:
   - Adaptive learning rate
   - Good for RNNs
   - Good for online learning
""")

if KERAS_AVAILABLE:
    # Different optimizers
    optimizers = {
        'Adam': Adam(learning_rate=0.001),
        'SGD': SGD(learning_rate=0.01),
        'Adam + LR=0.0001': Adam(learning_rate=0.0001)
    }

    print("\nComparing Optimizers:")
    print("-" * 40)

    for name, opt in optimizers.items():
        model_opt = Sequential([
            Dense(32, activation='relu', input_shape=(20,)),
            Dense(1, activation='sigmoid')
        ])
        model_opt.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model_opt.fit(X_train_scaled, y_train, epochs=10, verbose=0)
        _, acc = model_opt.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"{name}: {acc:.4f}")

print("\n" + "=" * 70)
print("KERAS BASICS SUMMARY")
print("=" * 70)

print("""
Key Takeaways:
1. Keras makes building NNs easy
2. Sequential API for simple architectures
3. Dense layers for fully connected networks
4. Use callbacks for better training
5. Always scale input data!
6. Dropout prevents overfitting

Next Steps:
- Learn CNN for image data
- Learn RNN for sequence data
- Explore transfer learning
""")
