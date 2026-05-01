"""
Neural Networks - Exercises
===========================

Practice problems for neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available")

# ============================================================================
# EXERCISE 1: Build MLP for MNIST
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Build MLP for MNIST")
print("=" * 70)

if TF_AVAILABLE:
    print("""
Task:
1. Load MNIST dataset
2. Flatten images to 1D
3. Build MLP with:
   - Dense(128, activation='relu')
   - Dense(64, activation='relu')
   - Dense(10, activation='softmax')
4. Train for 5 epochs
5. Evaluate accuracy
""")

    # Your code:
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten
    X_train_flat = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test_flat = X_test.reshape(-1, 784).astype('float32') / 255.0

    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(X_train_flat, y_train, epochs=5, verbose=1)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_flat, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

# ============================================================================
# EXERCISE 2: Experiment with Activation Functions
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Activation Functions")
print("=" * 70)

if TF_AVAILABLE:
    print("""
Task:
1. Build 3 models with different activations:
   - ReLU
   - Sigmoid
   - Tanh
2. Compare performance on same data
""")

    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    activations = ['relu', 'sigmoid', 'tanh']

    print("\nComparing Activation Functions:")
    for act in activations:
        model = Sequential([
            Dense(32, activation=act, input_shape=(20,)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=10, verbose=0)
        _, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"  {act}: {acc:.4f}")

# ============================================================================
# EXERCISE 3: Build CNN for Image Classification
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Build CNN")
print("=" * 70)

if TF_AVAILABLE:
    print("""
Task:
1. Build CNN with:
   - Conv2D(32, 3x3, relu)
   - MaxPooling2D
   - Conv2D(64, 3x3, relu)
   - Flatten
   - Dense(10, softmax)
2. Train on CIFAR-10 (or subset)
3. Evaluate
""")

    # Load CIFAR-10
    (X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = keras.datasets.cifar10.load_data()

    # Normalize
    X_train_cifar = X_train_cifar.astype('float32') / 255.0
    X_test_cifar = X_test_cifar.astype('float32') / 255.0

    # Build CNN
    cnn = Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train (small subset for demo)
    cnn.fit(X_train_cifar[:5000], y_train_cifar[:5000], epochs=2, verbose=1)

    # Evaluate
    loss, acc = cnn.evaluate(X_test_cifar, y_test_cifar)
    print(f"\nTest Accuracy: {acc:.4f}")

# ============================================================================
# EXERCISE 4: Add Dropout
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Dropout")
print("=" * 70)

if TF_AVAILABLE:
    print("""
Task:
1. Build model WITH dropout
2. Compare to model WITHOUT dropout
3. See the difference in overfitting
""")

    # Generate overfitting-prone data
    X, y = make_classification(n_samples=200, n_features=50, n_informative=20,
                              n_redundant=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Without dropout
    model_no_drop = Sequential([
        Dense(100, activation='relu', input_shape=(50,)),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_no_drop.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # With dropout
    model_drop = Sequential([
        Dense(100, activation='relu', input_shape=(50,)),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model_drop.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train both
    history_no_drop = model_no_drop.fit(X_train_scaled, y_train, epochs=50, verbose=0,
                                        validation_data=(X_test_scaled, y_test))
    history_drop = model_drop.fit(X_train_scaled, y_train, epochs=50, verbose=0,
                                  validation_data=(X_test_scaled, y_test))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history_no_drop.history['val_accuracy'], label='No Dropout')
    plt.plot(history_drop.history['val_accuracy'], label='With Dropout')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Effect of Dropout')
    plt.legend()
    plt.show()

    print(f"\nFinal Val Accuracy (No Dropout): {history_no_drop.history['val_accuracy'][-1]:.4f}")
    print(f"Final Val Accuracy (With Dropout): {history_drop.history['val_accuracy'][-1]:.4f}")

# ============================================================================
# EXERCISE 5: Build LSTM for Sequence
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: LSTM")
print("=" * 70)

if TF_AVAILABLE:
    from tensorflow.keras.layers import LSTM

    print("""
Task:
1. Create time series data
2. Build LSTM model
3. Make predictions
""")

    # Generate time series
    def generate_sine_data(n_samples=1000, seq_length=20):
        X, y = [], []
        for i in range(n_samples):
            start = np.random.uniform(0, 10)
            seq = [np.sin(start + j * 0.5) for j in range(seq_length)]
            target = np.sin(start + seq_length * 0.5)
            X.append(seq)
            y.append(target)
        return np.array(X), np.array(y)

    X_seq, y_seq = generate_sine_data()
    X_seq = X_seq.reshape(-1, seq_length, 1)

    split = int(0.8 * len(X_seq))
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

    # Build LSTM
    lstm = Sequential([
        LSTM(32, input_shape=(20, 1)),
        Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')

    lstm.fit(X_train_seq, y_train_seq, epochs=20, verbose=0)

    # Predict
    y_pred = lstm.predict(X_test_seq)
    mse = np.mean((y_test_seq - y_pred.flatten())**2)
    print(f"\nTest MSE: {mse:.4f}")

# ============================================================================
# EXERCISE 6: Use Early Stopping
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Early Stopping")
print("=" * 70)

if TF_AVAILABLE:
    print("""
Task:
1. Use EarlyStopping callback
2. Monitor val_loss
3. Restore best weights
""")

    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(X_train_scaled, y_train, epochs=50,
                        validation_split=0.2, callbacks=[early_stop], verbose=0)

    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")

# ============================================================================
# EXERCISE 7: Multi-class Classification
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Multi-class Classification")
print("=" * 70)

if TF_AVAILABLE:
    from tensorflow.keras.utils import to_categorical

    print("""
Task:
1. Load iris dataset
2. One-hot encode targets
3. Build and train model
4. Evaluate
""")

    # Load iris
    iris = load_iris()
    X, y = iris.data, iris.target

    # One-hot
    y_cat = to_categorical(y, 3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = Sequential([
        Dense(16, activation='relu', input_shape=(4,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_test, axis=1)

    print(f"\nAccuracy: {accuracy_score(y_true_class, y_pred_class):.4f}")

# ============================================================================
# BONUS: Custom Callback
# ============================================================================

print("\n" + "=" * 70)
print("BONUS: Custom Callback")
print("=" * 70)

if TF_AVAILABLE:
    print("""
Task:
1. Create custom callback
2. Print when accuracy exceeds threshold
""")

    class AccuracyThresholdCallback(keras.callbacks.Callback):
        def __init__(self, threshold=0.9):
            super().__init__()
            self.threshold = threshold

        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > self.threshold:
                print(f"\nReached {self.threshold*100}% accuracy at epoch {epoch+1}")

    # Use callback
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(20,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=10, callbacks=[AccuracyThresholdCallback(0.85)])

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
What you practiced:
1. Building MLP for MNIST
2. Comparing activation functions
3. Building CNN for images
4. Adding dropout to prevent overfitting
5. Using LSTM for sequences
6. Early stopping
7. Multi-class classification
8. Custom callbacks

Key Takeaways:
1. Start simple, add complexity
2. Use dropout for regularization
3. Choose activation based on task
4. Monitor validation performance
5. Use callbacks for better training
""")
