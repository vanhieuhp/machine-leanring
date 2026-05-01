# Computer Vision - Learning Guide

## What is Computer Vision?

Computer vision enables machines to interpret and understand visual information from the world, similar to how humans see and process images.

## Applications

| Application | Description | Examples |
|------------|-------------|----------|
| **Image Classification** | Assign labels to images | Photo organization, content moderation |
| **Object Detection** | Find and locate objects | Self-driving cars, surveillance |
| **Image Segmentation** | Pixel-level classification | Medical imaging, AR |
| **Face Recognition** | Identify faces | Security, social media |
| **Image Generation** | Create new images | Art, data augmentation |

## Learning Objectives

By the end of this section, you'll master:

### Fundamentals
1. **Image Processing** - Filtering, transformations, enhancement
2. **Convolutional Neural Networks** - CNN architecture
3. **Pooling & Padding** - Dimensionality reduction

### Advanced Techniques
1. **Transfer Learning** - Pre-trained models
2. **Object Detection** - YOLO, Faster R-CNN
3. **Image Segmentation** - U-Net, Mask R-CNN

### Modern Architectures
1. **ResNet** - Residual connections
2. **EfficientNet** - Compound scaling
3. **Vision Transformers** - Attention for images

## Key Concepts

### 1. CNN Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  CONVOLUTIONAL NEURAL NETWORK                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Image (H × W × 3)                                     │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────┐                   │
│  │  Conv2D + ReLU + BatchNorm           │  → Feature Maps  │
│  └──────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────┐                   │
│  │  MaxPool / AvgPool                   │  → Downsampling  │
│  └──────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────┐                   │
│  │  ... Repeat (Conv + Pool) ...       │  → Deep Features │
│  └──────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────┐                   │
│  │  Global Average Pooling / Flatten   │                   │
│  └──────────────────────────────────────┘                   │
│        │                                                     │
│        ▼                                                     │
│  ┌──────────────────────────────────────┐                   │
│  │  Dense Layers + Softmax              │  → Classification│
│  └──────────────────────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Convolutional Operation

```
Input:          Filter:           Output:
  ┌───┬───┬───┐     ┌───┬───┐    ┌─────┐
  │ 1 │ 2 │ 1 │     │ 1 │ 0 │    │  4  │
  ├───┼───┼───┤  *  ├───┼───┤ =  │ (1*1+2*1+1*0+...) │
  │ 0 │ 1 │ 2 │     │ 0 │ 1 │    │  4  │
  └───┴───┴───┘     └───┴───┘    └─────┘

Filter slides across input → Feature map
```

### 3. Transfer Learning

```
┌─────────────────────────────────────────────────────────────┐
│                  TRANSFER LEARNING                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Pre-trained Model (ImageNet):                              │
│  ┌─────────────────────────────────────────────┐           │
│  │  Conv blocks → Rich feature extraction     │           │
│  └─────────────────────────────────────────────┘           │
│                      │                                      │
│                      ▼                                      │
│  ┌─────────────────────────────────────────────┐           │
│  │  Custom classifier (your task)             │           │
│  │  Dense → Softmax                            │           │
│  └─────────────────────────────────────────────┘           │
│                                                              │
│  Strategy:                                                   │
│  1. Freeze base, train only classifier (fast)             │
│  2. Fine-tune entire network (slow, better)              │
│  3. Progressive unfreezing                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Data Augmentation

| Technique | Description |
|-----------|-------------|
| **Flip** | Horizontal/vertical flip |
| **Rotate** | Random rotation |
| **Crop** | Random crop and resize |
| **Color** | Brightness, contrast, saturation |
| **Noise** | Add Gaussian noise |
| **Cutout** | Random rectangular masks |

## Study Path

### Week 1: CNN Fundamentals
1. **Start with**: Image basics
   - NumPy image representation
   - Basic operations (resize, crop, normalize)

2. **Then**: Build CNN from scratch
   - Conv2D, MaxPool layers
   - Forward pass implementation

### Week 2: PyTorch/Keras for CV
3. **Next**: Deep Learning Frameworks
   - PyTorch: torch.nn, torchvision
   - Keras: Conv2D API

4. **Practice**: CIFAR-10 classification
   - Train from scratch
   - Understand overfitting

### Week 3: Transfer Learning
5. **Then**: Transfer Learning
   - Load pre-trained models
   - Fine-tune for custom dataset

### Week 4: Advanced Topics
6. **Finally**: Object Detection & Projects
   - YOLO for real-time detection
   - Image segmentation basics
   - Complete project

## Common Mistakes to Avoid

1. **Wrong image size** - Resize to model input size
2. **Not normalizing** - Use ImageNet mean/std for pre-trained models
3. **Data leakage** - Don't augment validation/test sets
4. **Learning rate** - Use lower LR for fine-tuning
5. **Overfitting** - Use dropout, data augmentation

## Framework Comparison

| Feature | PyTorch | Keras |
|---------|---------|-------|
| **Flexibility** | High | Medium |
| **Ease of use** | Medium | High |
| **Debugging** | Easy (Pythonic) | Harder |
| **Production** | TorchScript | TF Serving |
| **Research** | Popular | Popular |

---

**Difficulty**: Expert

**Estimated Time**: 1 month

**Prerequisites**: Phase 1 (NumPy), Phase 2 (Neural Networks)

**Next**: Recommendation Systems (Month 4)
