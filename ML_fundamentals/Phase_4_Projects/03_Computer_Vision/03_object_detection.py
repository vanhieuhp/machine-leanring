"""
Computer Vision - Part 3: Object Detection
=========================================

This module covers:
- Object detection concepts
- Bounding boxes
- YOLO basics
- Faster R-CNN
- Detection evaluation metrics
- Implementation with torchvision

Based on: Object Detection on Custom Images
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. OBJECT DETECTION CONCEPTS
# ============================================================================

print("=" * 70)
print("1. OBJECT DETECTION CONCEPTS")
print("=" * 70)

print("""
Object Detection vs Image Classification:
========================================

Image Classification:
  - Single label for entire image
  - "This is a cat"

Object Detection:
  - Find multiple objects
  - Localize each object with bounding box
  - "Cat at (x1,y1)-(x2,y2), Dog at (x3,y3)-(x4,y4)"

Localization + Classification = Object Detection

Bounding Box Format:
  - (x_min, y_min, x_max, y_max) - corner format
  - (x_center, y_center, width, height) - center format
  - Normalized (0-1) or pixel coordinates
""")

# Create sample bounding box
bbox = [50, 50, 150, 150]  # [x1, y1, x2, y2]
print(f"\nSample bounding box: {bbox}")
print(f"  x_min: {bbox[0]}, y_min: {bbox[1]}")
print(f"  x_max: {bbox[2]}, y_max: {bbox[3]}")
print(f"  Width: {bbox[2] - bbox[0]}")
print(f"  Height: {bbox[3] - bbox[1]}")

# ============================================================================
# 2. DETECTION ANNOTATIONS FORMAT
# ============================================================================

print("\n" + "=" * 70)
print("2. DETECTION ANNOTATIONS FORMAT")
print("=" * 70)

# COCO format
coco_annotation = {
    "images": [
        {
            "id": 1,
            "width": 640,
            "height": 480,
            "file_name": "image1.jpg"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,  # person
            "bbox": [100, 50, 50, 150],  # [x, y, w, h]
            "area": 7500,
            "iscrowd": 0
        }
    ],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "dog"}
    ]
}

print("COCO Format:")
print(f"  Image ID: {coco_annotation['images'][0]['id']}")
print(f"  Category: {coco_annotation['categories'][0]['name']}")
print(f"  BBox [x,y,w,h]: {coco_annotation['annotations'][0]['bbox']}")

# YOLO format (normalized center x, center y, width, height)
yolo_bbox = [0.234, 0.456, 0.123, 0.234]  # normalized
print(f"\nYOLO Format (normalized):")
print(f"  [center_x, center_y, width, height]")
print(f"  Values: {yolo_bbox}")

# Convert between formats
def box_xywh_to_xyxy(box):
    """Convert from (x_center, y_center, w, h) to (x1, y1, x2, y2)"""
    x_center, y_center, w, h = box
    x1 = x_center - w/2
    y1 = y_center - h/2
    x2 = x_center + w/2
    y2 = y_center + h/2
    return [x1, y1, x2, y2]

def box_xyxy_to_xywh(box):
    """Convert from (x1, y1, x2, y2) to (x_center, y_center, w, h)"""
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [x_center, y_center, w, h]

print("\nConversion functions:")
print(f"  xywh → xyxy: {box_xywh_to_xyxy([0.5, 0.5, 0.2, 0.2])}")
print(f"  xyxy → xywh: {box_xyxy_to_xywh([0.4, 0.4, 0.6, 0.6])}")

# ============================================================================
# 3. DETECTION EVALUATION METRICS
# ============================================================================

print("\n" + "=" * 70)
print("3. DETECTION EVALUATION METRICS")
print("=" * 70)

print("""
Key Metrics:
===========

1. IoU (Intersection over Union)
   - Measures overlap between predicted and ground truth boxes
   - IoU = Area of Intersection / Area of Union
   - IoU > threshold (usually 0.5) → True Positive

2. Precision & Recall
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - Higher is better

3. mAP (mean Average Precision)
   - Average precision across all classes
   - Primary metric for detection
   - mAP@0.5: IoU threshold 0.5
   - mAP@0.5:0.95: Average across IoU 0.5-0.95

4. FPS (Frames Per Second)
   - Inference speed
   - Real-time: >30 FPS
""")

# Calculate IoU
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Intersection area
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    iou = inter_area / union_area
    return iou

# Example IoU calculation
pred_box = [50, 50, 150, 150]
gt_box = [60, 60, 140, 140]

iou = calculate_iou(pred_box, gt_box)
print(f"\nIoU Calculation:")
print(f"  Predicted: {pred_box}")
print(f"  Ground Truth: {gt_box}")
print(f"  IoU: {iou:.4f}")

# IoU thresholds
print(f"\n  IoU > 0.5 → Detected (True Positive)")
print(f"  IoU ≤ 0.5 → Missed (False Positive)")

# ============================================================================
# 4. YOLO IMPLEMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("4. YOLO (YOU ONLY LOOK ONCE)")
print("=" * 70)

try:
    import torch
    import torchvision
    from torchvision.models.detection import yolo_resnet50, YOLO_RESNET50_CONFIG

    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    # Note: YOLO in torchvision might have different API
    # Using YOLOv5/YOLOv8 from ultralytics is more common

    print("\nYOLO Architecture:")
    print("  - Single forward pass")
    print("  - Grid-based detection")
    print("  - Anchor boxes")
    print("  - Non-maximum suppression (NMS)")

    # YOLO concepts
    print("\nYOLO Key Concepts:")
    print("  1. Divide image into S×S grid")
    print("  2. Each cell predicts B bounding boxes")
    print("  3. Each box has (x, y, w, h, confidence)")
    print("  4. Each cell predicts C class probabilities")
    print("  5. Total: S×S×(B×5 + C) outputs")

except Exception as e:
    print(f"Note: {e}")
    print("\nUsing YOLOv5/YOLOv8 from ultralytics is recommended:")
    print("  pip install ultralytics")

# Try YOLOv8
try:
    from ultralytics import YOLO

    print("\n" + "-" * 50)
    print("4.1 YOLOv8 Implementation")
    print("-" * 50)

    # Load model
    model = YOLO('yolov8n.pt')  # nano version
    print("YOLOv8n (nano) loaded")
    print("  - n: nano (fastest, least accurate)")
    print("  - s: small")
    print("  - m: medium")
    print("  - l: large")
    print("  - x: extra large")

    # Model info
    print(f"\nModel: yolov8n.pt")
    print("Classes: 80 (COCO)")

except ImportError:
    print("\nUltralytics not installed. Install with: pip install ultralytics")

# ============================================================================
# 5. FASTER R-CNN
# ============================================================================

print("\n" + "=" * 70)
print("5. FASTER R-CNN IMPLEMENTATION")
print("=" * 70)

try:
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.transforms import functional as F

    print("\n" + "-" * 50)
    print("5.1 Faster R-CNN Implementation")
    print("-" * 50)

    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    print("Faster R-CNN (ResNet50-FPN) loaded")
    print("  - Backbone: ResNet50 + FPN")
    print("  - RPN: Region Proposal Network")
    print("  - ROI Pooling: Align features")
    print("  - Detection head: Classification + BBox regression")

    # -------------------------------------------------------------------------
    # 5.2 Detection on Image
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("5.2 Running Detection")
    print("-" * 50)

    # Create dummy image (batch of 1)
    dummy_image = torch.rand(3, 640, 640)

    # Preprocessing (normalize)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (dummy_image - mean) / std

    # Inference
    with torch.no_grad():
        predictions = model([normalized])

    print(f"\nDetection results:")
    print(f"  Boxes: {predictions[0]['boxes'].shape[0]} detected")
    print(f"  Labels: {predictions[0]['labels'].shape}")
    print(f"  Scores: {predictions[0]['scores'].shape}")
    print(f"  Max score: {predictions[0]['scores'].max():.4f}")

except ImportError:
    print("PyTorch/Torchvision required for this section")

# ============================================================================
# 6. SSD (SINGLE SHOT DETECTOR)
# ============================================================================

print("\n" + "=" * 70)
print("6. SSD (SINGLE SHOT DETECTOR)")
print("=" * 70)

print("""
SSD vs Faster R-CNN:
====================

Faster R-CNN:
  - Two-stage detector
  - First: Generate proposals
  - Second: Refine and classify
  - More accurate, slower

SSD:
  - Single-shot detector
  - Direct classification and regression
  - Multi-scale feature maps
  - Faster, slightly less accurate

YOLO:
  - Similar to SSD
  - Latest versions (v5-v8) very accurate
  - Real-time capable
""")

# ============================================================================
# 7. NON-MAXIMUM SUPPRESSION (NMS)
# ============================================================================

print("\n" + "=" * 70)
print("7. NON-MAXIMUM SUPPRESSION (NMS)")
print("=" * 70)

print("""
NMS Purpose:
===========

After detection, we have many overlapping boxes.
NMS keeps the best one and removes others.

Algorithm:
=========
1. Sort boxes by confidence score
2. Pick highest confidence box
3. Remove all boxes with IoU > threshold
4. Repeat with remaining boxes

Parameters:
==========
- IoU threshold (usually 0.5)
- Score threshold (usually 0.5)
""")

# Implement NMS
def non_maximum_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.5):
    """
    Non-Maximum Suppression.

    Parameters:
    -----------
    boxes : array
        Bounding boxes [N, 4] in (x1, y1, x2, y2)
    scores : array
        Confidence scores [N]
    iou_threshold : float
        IoU threshold for suppression
    score_threshold : float
        Score threshold to keep

    Returns:
    --------
    keep : array
        Indices of boxes to keep
    """
    # Filter by score
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return []

    # Sort by score
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        # Pick highest score
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Calculate IoU with remaining boxes
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])

        # Keep boxes with IoU < threshold
        mask = ious < iou_threshold
        order = order[1:][mask]

    return keep

# Example
boxes = np.array([
    [50, 50, 150, 150],
    [55, 55, 145, 145],
    [200, 200, 300, 300],
    [210, 210, 290, 290],
    [50, 50, 100, 100]
])
scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.5)

print(f"\nNMS Example:")
print(f"  Input boxes: {len(boxes)}")
print(f"  Scores: {scores}")
print(f"  Kept indices: {keep}")
print(f"  Output boxes: {len(keep)}")

# ============================================================================
# 8. COMPLETE DETECTION PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("8. COMPLETE DETECTION PIPELINE")
print("=" * 70)

def detection_pipeline(image, model, score_threshold=0.5, iou_threshold=0.5):
    """
    Complete object detection pipeline.

    Steps:
    1. Preprocess image
    2. Run model
    3. Filter by score
    4. Apply NMS
    5. Return results
    """
    # 1. Preprocess
    # (resize, normalize, to tensor)

    # 2. Run model
    # predictions = model(image)

    # 3. Filter by score
    # mask = predictions['scores'] > score_threshold

    # 4. Apply NMS
    # keep = non_maximum_suppression(boxes, scores, iou_threshold)

    # 5. Return results
    # return boxes[keep], labels[keep], scores[keep]

    pass

# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("9. DETECTION MODEL COMPARISON")
print("=" * 70)

print("""
Model Comparison:
================

| Model          | mAP (COCO) | FPS  | Speed    | Best For       |
|----------------|------------|------|----------|----------------|
| YOLOv8-X       | 52-54%     | ~50  | Very Fast| Real-time     |
| YOLOv5-L       | 49%        | ~50  | Very Fast| Real-time     |
| Faster R-CNN   | 39%        | ~7   | Medium   | Accuracy      |
| SSD (MobileNet)| 22%        | ~20  | Fast     | Mobile        |
| RetinaNet      | 39%        | ~12  | Medium   | Balance       |

Real-time: >30 FPS
""")

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Object Detection
   - Find + localize + classify objects
   - Output: boxes, labels, scores

2. Bounding Boxes
   - Formats: xyxy, xywh (normalized)
   - Convert between formats as needed

3. Evaluation
   - IoU: Overlap measure
   - mAP: Mean Average Precision
   - Score threshold filters weak detections

4. Algorithms
   - Two-stage: Faster R-CNN (accurate)
   - One-stage: YOLO, SSD (fast)

5. NMS (Non-Maximum Suppression)
   - Removes duplicate detections
   - Keeps best box per object

6. Practical Tips
   - Use pre-trained models
   - Fine-tune on custom data
   - Consider speed vs accuracy trade-off

Next: exercises.py to practice
""")
