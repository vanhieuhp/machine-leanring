"""
Computer Vision - Part 2: Transfer Learning
=========================================

This module covers:
- Pre-trained models
- Feature extraction
- Fine-tuning
- Using torchvision and Keras Applications
- PyTorch implementation

Based on: Image Classification with Pre-trained Models
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. UNDERSTANDING TRANSFER LEARNING
# ============================================================================

print("=" * 70)
print("1. TRANSFER LEARNING CONCEPTS")
print("=" * 70)

print("""
Transfer Learning Overview:
=========================

Instead of training from scratch, we use pre-trained models that have
learned rich feature representations from large datasets (like ImageNet).

Why Transfer Learning?
----------------------
1. Limited data - Pre-trained models have seen millions of images
2. Computational cost - Training deep models takes days/weeks
3. Better generalization - Features transfer across tasks

Strategies:
-----------
1. Feature Extraction (Fast)
   - Freeze base layers
   - Train only classifier
   - Good for small datasets

2. Fine-tuning (Slower, often better)
   - Unfreeze some/all top layers
   - Train entire network
   - Better for larger datasets

3. Progressive Unfreezing
   - Unfreeze layers gradually
   - Start from top (closest to classifier)
""")

# ============================================================================
# 2. PYTORCH TRANSFER LEARNING
# ============================================================================

print("\n" + "=" * 70)
print("2. PYTORCH TRANSFER LEARNING")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    print(f"PyTorch version: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2.1 Load Pre-trained Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.1 Loading Pre-trained Models")
    print("-" * 50)

    # Load ResNet18
    print("\nLoading ResNet18...")
    resnet18 = models.resnet18(pretrained=True)

    # Modify for our number of classes
    num_classes = 10
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

    print("ResNet18 loaded and modified for 10 classes")

    # Show architecture
    print("\nResNet18 architecture (last layer):")
    print(f"  Original fc: {models.resnet18(pretrained=False).fc}")
    print(f"  New fc: {resnet18.fc}")

    # Load other models
    print("\nAvailable models:")
    print("  - ResNet: resnet18, resnet34, resnet50, resnet101")
    print("  - VGG: vgg11, vgg13, vgg16, vgg19")
    print("  - MobileNet: mobilenet_v2, mobilenet_v3_small")
    print("  - EfficientNet: efficientnet_b0, b1, ...")
    print("  - Inception: inception_v3")

    # -------------------------------------------------------------------------
    # 2.2 Feature Extraction Mode
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.2 Feature Extraction Mode")
    print("-" * 50)

    # Freeze all layers
    for param in resnet18.parameters():
        param.requires_grad = False

    # Only train the classifier
    for param in resnet18.fc.parameters():
        param.requires_grad = True

    # Count parameters
    frozen_params = sum(p.numel() for p in resnet18.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)

    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # 2.3 Fine-tuning Mode
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.3 Fine-tuning Mode")
    print("-" * 50)

    # Unfreeze all layers
    for param in resnet18.parameters():
        param.requires_grad = True

    # Or unfreeze specific layers
    # Unfreeze layers 4 and fc
    # Layers: conv1, bn1, layer1, layer2, layer3, layer4, avgpool, fc

    # Freeze early layers
    for name, param in resnet18.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    print("Fine-tuning strategy:")
    print("  - Frozen: conv1, bn1, layer1, layer2, layer3")
    print("  - Trainable: layer4, fc")

    # -------------------------------------------------------------------------
    # 2.4 Using Pre-trained Models for Feature Extraction
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.4 Feature Extraction")
    print("-" * 50)

    # Remove the final classifier
    feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
    feature_extractor.eval()

    # Extract features from dummy data
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = feature_extractor(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature dimension: {features.numel()}")

    # -------------------------------------------------------------------------
    # 2.5 Training with Transfer Learning
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.5 Training with Transfer Learning")
    print("-" * 50)

    # Create dummy data
    n_samples = 500
    X_train = torch.randn(n_samples, 3, 224, 224)
    y_train = torch.randint(0, 10, (n_samples,))

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Loss and optimizer (only for trainable parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Training
    print("Training for 2 epochs (demo)...")
    model.train()

    for epoch in range(2):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%")

    # -------------------------------------------------------------------------
    # 2.6 Image Transforms
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.6 Image Transforms")
    print("-" * 50)

    # ImageNet transforms
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    print("Training transforms:")
    print("  - Resize to 256")
    print("  - Random crop to 224")
    print("  - Random flip")
    print("  - Random rotation (±10°)")
    print("  - Color jitter")
    print("  - Normalize with ImageNet stats")

    print("\nValidation transforms:")
    print("  - Resize to 256")
    print("  - Center crop to 224")
    print("  - Normalize with ImageNet stats")

except ImportError:
    print("PyTorch not installed. Install with: pip install torch torchvision")

# ============================================================================
# 3. KERAS TRANSFER LEARNING
# ============================================================================

print("\n" + "=" * 70)
print("3. KERAS TRANSFER LEARNING")
print("=" * 70)

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing import image

    print(f"TensorFlow version: {tf.__version__}")

    # -------------------------------------------------------------------------
    # 3.1 Load Pre-trained Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.1 Loading Keras Pre-trained Models")
    print("-" * 50)

    # Load MobileNetV2 (lightweight, good for mobile)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    print("MobileNetV2 loaded")
    print(f"  Input shape: {base_model.input_shape}")
    print(f"  Output shape: {base_model.output_shape}")

    # -------------------------------------------------------------------------
    # 3.2 Build Transfer Learning Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.2 Building Transfer Learning Model")
    print("-" * 50)

    # Add classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    print("Transfer learning model built:")
    print("  - Base: MobileNetV2 (frozen)")
    print("  - GlobalAveragePooling2D")
    print("  - Dense(256, relu)")
    print("  - Dropout(0.5)")
    print("  - Dense(10, softmax)")

    # -------------------------------------------------------------------------
    # 3.3 Freeze Base Layers
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.3 Freezing Base Layers")
    print("-" * 50)

    # Freeze base model
    base_model.trainable = False

    # Count parameters
    frozen = len([w for w in model.trainable_weights if not w.trainable])
    trainable = len([w for w in model.trainable_weights if w.trainable])

    print(f"Frozen layers: {frozen}")
    print(f"Trainable layers: {trainable}")

    # -------------------------------------------------------------------------
    # 3.4 Fine-tuning
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.4 Fine-tuning Strategy")
    print("-" * 50)

    # Unfreeze last few layers
    base_model.trainable = True

    # Freeze all but last 10 layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Or unfreeze by name
    # for layer in base_model.layers:
    #     if 'block_15' in layer.name or 'block_16' in layer.name:
    #         layer.trainable = True

    print("Fine-tuning enabled for last 10 layers")

    # Recompile after unfreezing
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel recompiled with lower learning rate (0.0001)")

    # -------------------------------------------------------------------------
    # 3.5 Training
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.5 Training Transfer Learning Model")
    print("-" * 50)

    # Create dummy data
    X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, 100)

    # Phase 1: Train only classifier
    print("\nPhase 1: Training classifier only...")
    history1 = model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    # Phase 2: Fine-tune
    print("\nPhase 2: Fine-tuning...")
    history2 = model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

except ImportError as e:
    print(f"TensorFlow not available: {e}")

# ============================================================================
# 4. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("4. PRE-TRAINED MODEL COMPARISON")
print("=" * 70)

print("""
Model Comparison:
================

| Model       | Params (M) | Top-5 Acc | Speed  | Best For        |
|-------------|------------|-----------|--------|-----------------|
| MobileNetV2 | 3.5        | 96%       | Fast   | Mobile/Edge     |
| ResNet18    | 11.7       | 96%       | Medium | General purpose |
| ResNet50    | 25.6       | 97%       | Medium | High accuracy   |
| VGG16       | 138        | 95%       | Slow   | Feature extract |
| EfficientNet| 5.3M       | 98%       | Medium | Best accuracy   |
| InceptionV3 | 24         | 98%       | Medium | Complex images |

Selection Criteria:
-------------------
1. Dataset size: Large → can fine-tune more
2. Similarity to ImageNet: Similar → transfer works better
3. Computational budget: Limited → MobileNet
4. Accuracy requirements: High → EfficientNet/ResNet
""")

# ============================================================================
# 5. COMPLETE PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("5. COMPLETE TRANSFER LEARNING PIPELINE")
print("=" * 70)

def transfer_learning_pipeline(model_name='resnet18', num_classes=10,
                               train_data=None, train_labels=None,
                               fine_tune=False, epochs=10):
    """
    Complete transfer learning pipeline.

    Parameters:
    -----------
    model_name : str
        Name of pre-trained model
    num_classes : int
        Number of output classes
    train_data : array
        Training images
    train_labels : array
        Training labels
    fine_tune : bool
        Whether to fine-tune
    epochs : int
        Training epochs

    Returns:
    --------
    model, history
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        # Select model
        if model_name == 'resnet50':
            base = ResNet50(weights='imagenet', include_top=False)
        elif model_name == 'vgg16':
            base = VGG16(weights='imagenet', include_top=False)
        else:
            base = MobileNetV2(weights='imagenet', include_top=False)

        # Build model
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=predictions)

        # Freeze if not fine-tuning
        if not fine_tune:
            base.trainable = False

        # Compile
        lr = 0.001 if not fine_tune else 0.0001
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(
            train_data, train_labels,
            epochs=epochs,
            validation_split=0.1,
            verbose=0
        )

        return model, history

    except ImportError:
        print("TensorFlow required")
        return None, None

# Test pipeline
print("Testing transfer learning pipeline...")
try:
    X_dummy = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_dummy = np.random.randint(0, 10, 100)
    model, history = transfer_learning_pipeline(
        model_name='mobilenet',
        train_data=X_dummy,
        train_labels=y_dummy,
        epochs=2
    )
    if model:
        print("Pipeline completed successfully!")
except:
    print("Pipeline test skipped")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Why Transfer Learning?
   - Pre-trained models have learned powerful features
   - Works with limited data
   - Faster training

2. Feature Extraction
   - Freeze all base layers
   - Train only new classifier
   - Fast, good for small datasets

3. Fine-tuning
   - Unfreeze some/all layers
   - Use lower learning rate
   - Often gives better results

4. Best Practices
   - Use ImageNet normalization
   - Resize to expected input size (224x224)
   - Start with frozen, then unfreeze
   - Use lower LR for fine-tuning

5. Model Selection
   - MobileNet: Speed
   - ResNet: Balanced
   - EfficientNet: Accuracy

Next: Object Detection (03_object_detection.py)
""")
