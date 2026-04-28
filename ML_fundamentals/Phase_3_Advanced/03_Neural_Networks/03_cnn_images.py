"""
=================================================================
03 - CNN: Convolutional Neural Networks for Image Classification
=================================================================
Topics:
  1. CNN building blocks (Conv2D, MaxPool, Flatten)
  2. MNIST digit classification
  3. Fashion-MNIST classification
  4. Data augmentation
  5. Transfer learning concepts
  6. CNN architecture patterns
=================================================================
Prerequisites: pip install tensorflow
=================================================================
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("❌ TensorFlow not installed. Run: pip install tensorflow")

if not HAS_TF:
    print("\n⚠️ This file requires TensorFlow. Showing concepts only.\n")

# ── Section 1: CNN Building Blocks ────────────────────────────────
print("=" * 65)
print("SECTION 1: CNN Building Blocks")
print("=" * 65)

print("""
  🧱 Conv2D: Convolutional Layer
    • Slides filters (kernels) across the image
    • Each filter detects a specific pattern (edge, texture, etc.)
    • Output: feature map

    Input: 28×28×1  →  Conv2D(32, 3×3)  →  Output: 26×26×32
                        (32 filters)

  🧱 MaxPool2D: Pooling Layer
    • Downsamples feature maps (reduces spatial size)
    • Takes max value in each window

    Input: 26×26×32  →  MaxPool2D(2×2)  →  Output: 13×13×32

  🧱 Flatten: Reshape to 1D
    • Converts 2D feature maps to 1D vector for Dense layers

    Input: 13×13×32  →  Flatten()  →  Output: 5408

  🧱 Dense: Fully Connected Layer
    • Standard neural network layer for classification
""")

# ── Section 2: MNIST Digit Classification ─────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: MNIST Digit Classification")
print("=" * 65)

if HAS_TF:
    # Load MNIST
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"\n  Dataset shape: {X_train.shape}")
    print(f"  Image size: {X_train.shape[1]}×{X_train.shape[2]} pixels")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Classes: {np.unique(y_train)} (digits 0-9)")

    # Preprocess: normalize to [0, 1] and reshape for CNN
    X_train_cnn = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test_cnn = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # --- 2.1 Simple CNN ---
    print("\n📌 2.1 Simple CNN Architecture:")
    model_simple = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])

    model_simple.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    print(f"\n  Model architecture:")
    model_simple.summary()

    print("\n  Training (5 epochs)...")
    hist = model_simple.fit(
        X_train_cnn, y_train, epochs=5, batch_size=64,
        validation_split=0.1, verbose=1,
    )

    test_loss, test_acc = model_simple.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"\n  ✅ Test Accuracy: {test_acc:.4f}")
    print(f"  ✅ That's {test_acc * 100:.1f}% correct on 10,000 test images!")

    # --- 2.2 Better CNN (with BatchNorm + Dropout) ---
    print("\n\n📌 2.2 Improved CNN (BatchNorm + Dropout):")
    model_better = keras.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])

    model_better.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    hist2 = model_better.fit(
        X_train_cnn, y_train, epochs=5, batch_size=64,
        validation_split=0.1, verbose=1,
    )

    test_loss2, test_acc2 = model_better.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"\n  Simple CNN:   {test_acc:.4f}")
    print(f"  Improved CNN: {test_acc2:.4f}")

# ── Section 3: Fashion-MNIST ─────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: Fashion-MNIST (Harder Problem)")
print("=" * 65)

if HAS_TF:
    (X_f_train, y_f_train), (X_f_test, y_f_test) = keras.datasets.fashion_mnist.load_data()
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    X_f_train_cnn = X_f_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_f_test_cnn = X_f_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print(f"\n  Classes: {class_names}")

    model_fashion = keras.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax'),
    ])

    model_fashion.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("  Training...")
    hist_f = model_fashion.fit(
        X_f_train_cnn, y_f_train, epochs=15, batch_size=64,
        validation_split=0.1, callbacks=[early_stop], verbose=0,
    )

    _, f_acc = model_fashion.evaluate(X_f_test_cnn, y_f_test, verbose=0)
    print(f"  Fashion-MNIST Test Accuracy: {f_acc:.4f}")

    # Per-class accuracy
    y_pred_f = model_fashion.predict(X_f_test_cnn, verbose=0).argmax(axis=1)
    print(f"\n  Per-class accuracy:")
    for i, name in enumerate(class_names):
        mask = y_f_test == i
        class_acc = (y_pred_f[mask] == y_f_test[mask]).mean()
        print(f"    {name:>12s}: {class_acc:.4f}")

# ── Section 4: Data Augmentation ──────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Data Augmentation")
print("=" * 65)

print("""
  Data Augmentation creates variations of training images:
    • Random rotation, flip, zoom, shift
    • Increases effective dataset size
    • Reduces overfitting
    • ONLY applied during training (not test)

  Common augmentations:
    ┌────────────────────┬──────────────────────────────┐
    │ Augmentation       │ Keras Layer                  │
    ├────────────────────┼──────────────────────────────┤
    │ Random flip        │ RandomFlip("horizontal")     │
    │ Random rotation    │ RandomRotation(0.1)          │
    │ Random zoom        │ RandomZoom(0.1)              │
    │ Random translation │ RandomTranslation(0.1, 0.1)  │
    │ Random contrast    │ RandomContrast(0.1)          │
    └────────────────────┴──────────────────────────────┘
""")

if HAS_TF:
    # Create augmentation pipeline
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Model with augmentation
    model_aug = keras.Sequential([
        # Augmentation (only active during training)
        data_augmentation,
        # CNN layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax'),
    ])

    model_aug.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("  Training with augmentation...")
    hist_aug = model_aug.fit(X_f_train_cnn, y_f_train, epochs=10, batch_size=64,
                            validation_split=0.1, verbose=0)
    _, aug_acc = model_aug.evaluate(X_f_test_cnn, y_f_test, verbose=0)
    print(f"  Without augmentation: {f_acc:.4f}")
    print(f"  With augmentation:    {aug_acc:.4f}")

# ── Section 5: Architecture Patterns ──────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: CNN Architecture Patterns")
print("=" * 65)

print("""
  📋 Common CNN Architecture Patterns:

  1. LeNet-5 (1998) — Simple, for small images
     Conv→Pool→Conv→Pool→FC→FC→Output

  2. VGG (2014) — Deep, uniform architecture
     [Conv×2→Pool]×5→FC→FC→Output

  3. ResNet (2015) — Skip connections, very deep
     Conv→[ResBlock]×N→GlobalPool→FC
     Each ResBlock: x + F(x) (skip connection)

  4. Modern Pattern:
     Conv→BN→ReLU→Pool → [Conv→BN→ReLU]×N → GlobalAvgPool → Dense

  📏 Rules of Thumb:
    • Double filters as spatial dims halve: 32→64→128→256
    • Use 3×3 convolutions (two 3×3 ≈ one 5×5)
    • GlobalAveragePooling instead of Flatten (fewer params)
    • BN after Conv, before activation
""")

if HAS_TF:
    # Modern architecture with GlobalAveragePooling
    model_modern = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),  # Instead of Flatten
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax'),
    ])

    model_modern.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"\n  Total parameters: {model_modern.count_params():,}")
    print("  Training modern architecture...")
    model_modern.fit(X_f_train_cnn, y_f_train, epochs=10, batch_size=64,
                     validation_split=0.1, verbose=0)
    _, modern_acc = model_modern.evaluate(X_f_test_cnn, y_f_test, verbose=0)
    print(f"  Modern CNN accuracy: {modern_acc:.4f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
✅ Key Takeaways:
  1. CNN = Conv2D + MaxPool + Flatten + Dense
  2. Conv2D extracts local features, Pooling downsamples
  3. BatchNorm + Dropout are essential for good performance
  4. Data augmentation helps prevent overfitting
  5. Use GlobalAveragePooling for fewer parameters
  6. Double filters as spatial dimensions decrease
  7. For real tasks: use pretrained models (transfer learning)

📊 Typical CNN Pipeline:
  1. Load & preprocess images (normalize to [0,1])
  2. Build CNN architecture
  3. Compile (adam + categorical_crossentropy)
  4. Train with augmentation + early stopping
  5. Evaluate & iterate

📚 Next: 04_rnn_sequences.py (RNN/LSTM for Sequential Data)
""")
