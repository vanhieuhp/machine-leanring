"""
Computer Vision - Practice Exercises
==================================

Complete these exercises to solidify your computer vision skills.
Solutions are provided at the bottom.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXERCISE 1: Image Representation
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Image Representation")
print("=" * 70)

# 1.1 Create a random RGB image
# TODO: Create a 64x64 RGB image with values 0-255
# HINT: Use np.random.randint

image = None  # TODO: Create image

print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Value range: [{image.min()}, {image.max()}]")

# 1.2 Normalize image to [0, 1]
# TODO: Normalize
image_normalized = None  # TODO: Normalize

print(f"\nNormalized range: [{image_normalized.min():.2f}, {image_normalized.max():.2f}]")

# ============================================================================
# EXERCISE 2: Convolution
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Convolution")
print("=" * 70)

# 2.1 Create edge detection kernel
# TODO: Create a 3x3 Sobel edge detection kernel
edge_kernel = None  # TODO: Create

# 2.2 Create blur kernel (average)
# TODO: Create 3x3 blur kernel
blur_kernel = None  # TODO: Create

print(f"Edge kernel:\n{edge_kernel}")
print(f"\nBlur kernel:\n{blur_kernel}")

# ============================================================================
# EXERCISE 3: CNN Model
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Build CNN Model")
print("=" * 70)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    # 3.1 Build a CNN with:
    # - Conv2D(32, 3x3, relu, input_shape=(32,32,3))
    # - MaxPooling2D(2x2)
    # - Conv2D(64, 3x3, relu)
    # - MaxPooling2D(2x2)
    # - Flatten
    # - Dense(128, relu)
    # - Dropout(0.5)
    # - Dense(10, softmax)

    model = None  # TODO: Build

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    print("Model summary:")
    model.summary()

except ImportError:
    print("TensorFlow not available")

# ============================================================================
# EXERCISE 4: Data Augmentation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Data Augmentation")
print("=" * 70)

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # 4.1 Create augmentation with:
    # - rotation_range=20
    # - width_shift_range=0.2
    # - height_shift_range=0.2
    # - horizontal_flip=True
    # - zoom_range=0.2

    datagen = None  # TODO: Create

    print("Data augmentation configured:")
    print("  - Rotation: 20°")
    print("  - Width shift: 20%")
    print("  - Height shift: 20%")
    print("  - Horizontal flip")
    print("  - Zoom: 20%")

except ImportError:
    print("TensorFlow required")

# ============================================================================
# EXERCISE 5: Transfer Learning
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Transfer Learning")
print("=" * 70)

try:
    from tensorflow.keras.applications import VGG16

    # 5.1 Load VGG16 without top
    # TODO: Load with weights='imagenet', include_top=False
    base_model = None  # TODO: Load

    print(f"Base model input: {base_model.input_shape}")
    print(f"Base model output: {base_model.output_shape}")

    # 5.2 Freeze the model
    # TODO: Set trainable = False
    base_model.trainable = None  # TODO: Freeze

    print(f"Trainable after freeze: {base_model.trainable}")

except ImportError:
    print("TensorFlow required")

# ============================================================================
# EXERCISE 6: Bounding Box Conversion
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Bounding Box Operations")
print("=" * 70)

# 6.1 Convert xywh to xyxy
# TODO: Implement box_xywh_to_xyxy([x_center, y_center, w, h])
def box_xywh_to_xyxy(box):
    """Convert center format to corner format."""
    # TODO: Implement
    return [0, 0, 0, 0]

# 6.2 Calculate IoU
# TODO: Implement calculate_iou(box1, box2)
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    # TODO: Implement
    return 0.0

# Test
box1 = [10, 10, 50, 50]  # [x1, y1, x2, y2]
box2 = [30, 30, 70, 70]

result = calculate_iou(box1, box2)
print(f"IoU of {box1} and {box2}: {result:.4f}")

# ============================================================================
# EXERCISE 7: NMS Implementation
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: Non-Maximum Suppression")
print("=" * 70)

# 7.1 Implement NMS
def nms(boxes, scores, iou_threshold=0.5):
    """
    Implement Non-Maximum Suppression.
    """
    # TODO: Implement
    # 1. Sort by scores
    # 2. Pick highest, remove overlapping
    # 3. Repeat
    return []

# Test
boxes = np.array([
    [10, 10, 50, 50],
    [15, 15, 45, 45],
    [100, 100, 200, 200]
])
scores = np.array([0.9, 0.8, 0.7])

keep = nms(boxes, scores)
print(f"Boxes: {len(boxes)}, Kept: {keep}")

# ============================================================================
# EXERCISE 8: Pre-trained Model Loading
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Pre-trained Models")
print("=" * 70)

try:
    import torchvision.models as models

    # 8.1 Load ResNet18
    # TODO: Load pretrained ResNet18
    resnet = None  # TODO: Load

    print("ResNet18 loaded successfully")

    # 8.2 Count parameters
    # TODO: Count total and trainable parameters
    total = 0  # TODO: Count
    trainable = 0  # TODO: Count

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

except ImportError:
    print("PyTorch required")

# ============================================================================
# EXERCISE 9: Image Transforms
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9: Image Transforms")
print("=" * 70)

try:
    import torchvision.transforms as transforms

    # 9.1 Create training transforms
    # TODO: Create Compose with:
    # - Resize(256)
    # - RandomResizedCrop(224)
    # - RandomHorizontalFlip()
    # - ToTensor()
    # - Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = None  # TODO: Create

    # 9.2 Create validation transforms
    # TODO: Create Compose with:
    # - Resize(256)
    # - CenterCrop(224)
    # - ToTensor()
    # - Normalize(...)

    val_transforms = None  # TODO: Create

    print("Transforms created:")
    print("  Training: Resize → RandomCrop → Flip → ToTensor → Normalize")
    print("  Validation: Resize → CenterCrop → ToTensor → Normalize")

except ImportError:
    print("PyTorch required")

# ============================================================================
# EXERCISE 10: Complete Detection Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 10: Complete Detection Pipeline")
print("=" * 70)

def detect_objects(image, model, score_threshold=0.5):
    """
    Complete object detection pipeline.

    Steps:
    1. Preprocess image
    2. Run model
    3. Filter by score
    4. Apply NMS
    5. Return results

    Parameters:
    -----------
    image : tensor
        Input image
    model : nn.Module
        Detection model
    score_threshold : float
        Minimum score

    Returns:
    --------
    boxes, labels, scores
    """
    # TODO: Implement

    return [], [], []

# ============================================================================
# SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTIONS")
print("=" * 70)

print("""
EXERCISE 1:
1.1: image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
1.2: image_normalized = image / 255.0

EXERCISE 2:
2.1: edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
2.2: blur_kernel = np.ones((3, 3)) / 9

EXERCISE 3:
3.1: model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(10, activation='softmax')
  ])

EXERCISE 4:
4.1: datagen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      zoom_range=0.2
  )

EXERCISE 5:
5.1: base_model = VGG16(weights='imagenet', include_top=False)
5.2: base_model.trainable = False

EXERCISE 6:
6.1: def box_xywh_to_xyxy(box):
      x_c, y_c, w, h = box
      return [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]

6.2: def calculate_iou(box1, box2):
      x1 = max(box1[0], box2[0])
      y1 = max(box1[1], box2[1])
      x2 = min(box1[2], box2[2])
      y2 = min(box1[3], box2[3])
      inter = max(0, x2-x1) * max(0, y2-y1)
      area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
      area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
      return inter / (area1 + area2 - inter)

EXERCISE 7:
7.1: def nms(boxes, scores, iou_threshold=0.5):
      order = scores.argsort()[::-1]
      keep = []
      while len(order) > 0:
          i = order[0]
          keep.append(i)
          if len(order) == 1: break
          ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])
          order = order[1:][ious < iou_threshold]
      return keep

EXERCISE 8:
8.1: resnet = models.resnet18(pretrained=True)
8.2: total = sum(p.numel() for p in resnet.parameters())
      trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)

EXERCISE 9:
9.1: train_transforms = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

9.2: val_transforms = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

EXERCISE 10:
10.1: Implement full detection pipeline with preprocessing, inference,
      score filtering, and NMS
""")
