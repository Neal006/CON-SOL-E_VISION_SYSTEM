# SPGS-Net: Prior-Guided Segmentation Network for Multi-Defect Detection on Industrial Surfaces

## Technical Documentation for Journal Publication

---

## Abstract

This document presents SPGS-Net (Prior-Guided Segmentation Network), a novel deep learning architecture for automated multi-class defect detection on industrial surfaces. The system combines self-supervised visual representations from DINOv2, classical machine learning via XGBoost for spatial prior generation, and an Attention U-Net for pixel-precise segmentation. The architecture addresses key challenges in industrial defect detection: limited labeled data, class imbalance, and the need for precise defect localization with real-world area estimation.

---

## 1. Introduction and Problem Statement

### 1.1 Industrial Defect Detection Challenges

| Challenge | Description | Our Solution |
|-----------|-------------|--------------|
| Limited labeled data | Industrial datasets typically have <1000 images | Leverage frozen DINOv2 pretrained on 142M images |
| Class imbalance | Background >> Defects; Dust >> Scratch | Focal Loss + Prior-based reweighting |
| Diverse defect morphology | Scratches (thin, linear) vs Dust (blob-like) | Per-class detection thresholds |
| Precise localization | Need pixel-level masks, not just bounding boxes | Attention U-Net with skip connections |
| Physical measurements | Factory systems need area in mm², not pixels | Camera calibration integration |

### 1.2 Dataset Characteristics

- **Total Images:** 970 training, 55 validation, 28 test
- **Image Resolution:** 1440 × 1080 pixels (industrial camera)
- **Classes:** 3 defect types + background
  - Class 0: Background (clean surface)
  - Class 1: Dust (particulate contamination)
  - Class 2: RunDown (coating irregularities)
  - Class 3: Scratch (linear surface damage)
- **Annotation Format:** YOLO polygon format (normalized coordinates)

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SPGS-Net Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Section 1  │     │   Section 2  │     │   Section 3  │                 │
│  │    Input     │────▶│   DINOv2     │────▶│   XGBoost    │                 │
│  │ Preprocessing│     │  (Frozen)    │     │  Classifier  │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                                          │                         │
│         │                                          ▼                         │
│         │                                  ┌──────────────┐                  │
│         │                                  │   Section 4  │                  │
│         │                                  │  Upsampling  │                  │
│         │                                  └──────────────┘                  │
│         │                                          │                         │
│         │              Anomaly Prior               │                         │
│         │         ┌────────────────────────────────┘                         │
│         ▼         ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐                 │
│  │                     Section 5                            │                 │
│  │              Attention U-Net Segmentation                │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Section 6  │     │   Section 7  │     │   Section 8  │                 │
│  │    Instance  │────▶│     Area     │────▶│    Output    │                 │
│  │  Separation  │     │  Estimation  │     │ Visualization│                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Section 1: Input Acquisition & Preprocessing

### 3.1 Image Loading Pipeline

```python
Input: Raw industrial camera image (1440 × 1080 × 3, BGR)
       │
       ▼
Step 1: BGR → RGB conversion (OpenCV default is BGR)
       │
       ▼
Step 2: Grayscale → 3-channel replication (if grayscale input)
       │
       ▼
Step 3: Resize to training size (560 × 560) - divisible by 14
       │
       ▼
Step 4: ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
       │
       ▼
Output: Normalized tensor (3, 560, 560)
```

### 3.2 Label Processing

YOLO polygon format conversion:
```
Input:  "3 0.123 0.456 0.234 0.567 0.345 0.678 ..."
        │
        └── Class ID (3 = Scratch)
            └── Normalized polygon coordinates (x1,y1,x2,y2,...)

Processing:
1. Parse class ID and coordinates
2. Denormalize: pixel_x = norm_x × image_width
3. Create binary polygon mask using cv2.fillPoly()
4. Combine into multi-class segmentation mask (H, W)
   - Values: 0=Background, 1=Dust, 2=RunDown, 3=Scratch
```

### 3.3 Data Augmentation (Training Only)

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| HorizontalFlip | p=0.5 | Orientation invariance |
| VerticalFlip | p=0.5 | Orientation invariance |
| RandomRotate90 | p=0.5 | Rotation invariance |
| RandomBrightnessContrast | ±20%, p=0.5 | Lighting variation |
| GaussNoise | var=(10, 50), p=0.3 | Sensor noise simulation |

---

## 4. Section 2: DINOv2 Feature Extraction

### 4.1 DINOv2 Architecture Selection

We employ **DINOv2-ViT-S/14** (Vision Transformer Small with 14×14 patches):

| Component | Specification |
|-----------|---------------|
| Model | `dinov2_vits14` |
| Patch Size | 14 × 14 pixels |
| Hidden Dimension | 384 |
| Number of Layers | 12 |
| Attention Heads | 6 |
| Parameters | 21M (frozen) |
| Pretraining | Self-supervised on LVD-142M (142 million images) |

### 4.2 Vision Transformer Architecture

```
Input Image: (B, 3, H, W) where H=W=560

Step 1: Patch Embedding
        560 ÷ 14 = 40 patches per dimension
        Total patches = 40 × 40 = 1600 patches
        Each patch: 14 × 14 × 3 = 588 pixels → Linear projection → 384 dims

Step 2: Position Embedding
        Learnable position embeddings added to each patch
        Shape: (1600, 384)

Step 3: [CLS] Token Prepending
        Shape becomes: (1 + 1600, 384) = (1601, 384)

Step 4: Transformer Encoder (12 layers)
        Each layer:
        ┌────────────────────────────────────────┐
        │  LayerNorm                              │
        │      ↓                                  │
        │  Multi-Head Self-Attention (6 heads)   │
        │      ↓                                  │
        │  Residual Connection                    │
        │      ↓                                  │
        │  LayerNorm                              │
        │      ↓                                  │
        │  MLP (384 → 1536 → 384)                │
        │      ↓                                  │
        │  Residual Connection                    │
        └────────────────────────────────────────┘

Output: (B, 1601, 384) - [CLS] token + 1600 patch tokens
```

### 4.3 Multi-Layer Feature Extraction

We extract features from three transformer layers to capture different levels of abstraction:

| Layer | Index | Feature Type | Characteristics |
|-------|-------|--------------|-----------------|
| Low-level | 3 | Texture features | Edge patterns, local texture variations |
| Mid-level | 6 | Structural features | Shape patterns, structural anomalies |
| High-level | 11 | Semantic features | Contextual understanding, global patterns |

**Feature Aggregation:**
```
Layer 3 output: (B, 1600, 384)
Layer 6 output: (B, 1600, 384)
Layer 11 output: (B, 1600, 384)
        │
        ▼
Concatenation: (B, 1600, 1152)  # 384 × 3
        │
        ▼
Linear Projection: (B, 1600, 384)
        │
        ▼
Reshape to spatial: (B, 384, 40, 40)
```

### 4.4 Rationale for DINOv2

1. **Self-supervised pretraining**: No labels required during backbone training
2. **Texture sensitivity**: DINO objectives create features sensitive to visual differences
3. **Transfer learning**: 142M image pretraining provides robust representations
4. **Frozen backbone**: Prevents overfitting on small industrial datasets
5. **Patch-level features**: Natural fit for spatial anomaly localization

---

## 5. Section 3: XGBoost Patch Classifier

### 5.1 Problem Formulation

Convert dense DINOv2 features into a spatial anomaly heatmap:

```
Input: Patch features (B, 384, 40, 40)
       Reshape to: (B × 1600, 384) = (N, 384) samples

Output: Per-patch class probabilities (N, 4)
        Classes: [Background, Dust, RunDown, Scratch]
```

### 5.2 XGBoost Configuration

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| n_estimators | 200 | Sufficient for convergence with early stopping |
| max_depth | 6 | Prevent overfitting, capture reasonable interactions |
| learning_rate | 0.1 | Standard value, with early stopping |
| min_child_weight | 1 | Default, allow small leaf nodes |
| subsample | 0.8 | Row subsampling for regularization |
| colsample_bytree | 0.8 | Feature subsampling for regularization |
| objective | multi:softprob | Multi-class probability output |
| eval_metric | mlogloss | Multi-class logarithmic loss |
| early_stopping_rounds | 20 | Prevent overfitting |

### 5.3 Patch Label Generation

Critical for training quality. We use **per-class thresholds** for labeling:

| Class | Threshold | Rationale |
|-------|-----------|-----------|
| Dust | 10% | Blob-like defects occupy significant patch area |
| RunDown | 10% | Similar to dust in spatial extent |
| Scratch | 1% | Thin linear defects may only cover 2-3 pixels per patch |

**Priority Order:** Scratch > RunDown > Dust  
If multiple defects present in a patch, prioritize thin/hard-to-detect classes.

### 5.4 Anomaly Score Computation

```python
# Per-patch probabilities from XGBoost
P(class | patch) = [P(bg), P(dust), P(rundown), P(scratch)]

# Anomaly score = probability of ANY defect
anomaly_score = 1 - P(background)
              = P(dust) + P(rundown) + P(scratch)

# Reshape to spatial grid
heatmap = anomaly_scores.reshape(40, 40)  # For 560×560 input

# Normalize to [0, 1]
heatmap = (heatmap - min) / (max - min)
```

---

## 6. Section 4: Anomaly Prior Upsampling

### 6.1 Upsampling Strategy

```
Input: Patch-level heatmap (B, 1, 40, 40)
        │
        ▼
Bilinear Interpolation (align_corners=True)
        │
        ▼
Output: Full-resolution prior (B, 1, 560, 560)
```

### 6.2 Optional Gaussian Smoothing

To reduce patch boundary artifacts:
```
Kernel size: 5×5
Sigma: 1.0
Application: 2D convolution after upsampling
```

### 6.3 Mathematical Formulation

For bilinear interpolation at position (x, y):
```
f(x, y) = f(x₁, y₁)(x₂-x)(y₂-y) + f(x₂, y₁)(x-x₁)(y₂-y)
        + f(x₁, y₂)(x₂-x)(y-y₁) + f(x₂, y₂)(x-x₁)(y-y₁)
```
Where (x₁, y₁), (x₂, y₂) are the four nearest patch centers.

---

## 7. Section 5: Attention U-Net Architecture

### 7.1 Overall Architecture

```
ENCODER                           DECODER
────────                          ────────
Conv Block 1 (3→64)         ───── Up Block 1 (128→64) ─────── Output
    │                         ↑        ↑
    ▼                         │   Attention Gate
MaxPool                       │        │
    │                         │        │
Conv Block 2 (64→128)       ───── Up Block 2 (256→128)
    │                         ↑        ↑
    ▼                         │   Attention Gate
MaxPool                       │        │
    │                         │        │
Conv Block 3 (128→256)      ───── Up Block 3 (512→256)
    │                         ↑        ↑
    ▼                         │   Attention Gate
MaxPool                       │        │
    │                         │        │
Conv Block 4 (256→512)      ───── Up Block 4 (1024→512)
    │                         ↑        ↑
    ▼                         │   Attention Gate
MaxPool                       │        │
    │                         │        │
    └──────▶ BOTTLENECK (512→1024) ────┘
```

### 7.2 Convolution Block (Double Conv)

```
Input: (B, C_in, H, W)
    │
    ▼
Conv2d(C_in, C_out, kernel=3, padding=1)
    │
    ▼
BatchNorm2d(C_out)
    │
    ▼
ReLU(inplace=True)
    │
    ▼
Dropout2d(p=0.3)  [training only]
    │
    ▼
Conv2d(C_out, C_out, kernel=3, padding=1)
    │
    ▼
BatchNorm2d(C_out)
    │
    ▼
ReLU(inplace=True)
    │
    ▼
Output: (B, C_out, H, W)
```

### 7.3 Attention Gate Mechanism

The key innovation for prior injection:

```
Skip connection (x): (B, C, H, W) - from encoder
Gating signal (g):   (B, C', H', W') - from decoder (upsampled)
Prior (optional):    (B, 1, H, W) - anomaly prior map

Step 1: Project both to intermediate dimension
        W_x(x) → (B, C_int, H, W)
        W_g(g) → (B, C_int, H, W)  [after upsampling to match x]

Step 2: Combine and activate
        ψ = sigmoid(Conv1x1(ReLU(W_x + W_g)))
        Shape: (B, 1, H, W) - attention coefficients

Step 3: Prior modulation (if prior available)
        ψ_enhanced = ψ × (1 + prior)
        Clamp to [0, 1]

Step 4: Apply attention to skip features
        output = x × ψ_enhanced
```

**Interpretation:** The attention gate learns which spatial locations in the skip connection are relevant for reconstruction. The prior biases attention toward defect-likely regions.

### 7.4 Channel Progression

| Stage | Encoder | Decoder | Resolution (560×560 input) |
|-------|---------|---------|---------------------------|
| 1 | 64 | 64 | 560 × 560 |
| 2 | 128 | 128 | 280 × 280 |
| 3 | 256 | 256 | 140 × 140 |
| 4 | 512 | 512 | 70 × 70 |
| Bottleneck | 1024 | - | 35 × 35 |

### 7.5 Output Layer

```
Final Conv: Conv2d(64, 4, kernel=1)
Output: (B, 4, H, W) - logits for 4 classes
```

---

## 8. Section 9: Training Strategy

### 8.1 Loss Function Design

**Combined Loss = α × Dice Loss + β × Focal Loss**

Where α = β = 0.5 (equal weighting)

#### Dice Loss (Overlap-based)

```
Dice(p, g) = 2 × |p ∩ g| / (|p| + |g|)

Dice Loss = 1 - Dice

For each class c:
    intersection = Σ(p_c × g_c)
    union = Σ(p_c) + Σ(g_c)
    dice_c = (2 × intersection + ε) / (union + ε)

Total Dice Loss = mean(1 - dice_c) for all classes
```

#### Focal Loss (Class Imbalance)

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

Parameters:
    γ = 2.0 (focusing parameter)
    α = 0.25 (class balance weight)

Effect: Down-weights well-classified examples
        Up-weights hard-to-classify examples (defects)
```

### 8.2 Prior-Weighted Loss (Optional)

```
For each pixel (i, j):
    weight(i,j) = 1 + λ × prior(i,j)
    
Where λ = 2.0 (prior reweight factor)

Effect: Pixels with high anomaly prior get 3× loss weight
        Focuses learning on defect-likely regions
```

### 8.3 Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Initial LR | 1e-4 |
| Weight Decay | 1e-5 |
| LR Schedule | Cosine Annealing |
| Epochs | 50-100 |
| Batch Size | 4 |
| Early Stopping | Patience = 15 |

### 8.4 Training Pipeline

```
For each epoch:
    For each batch (images, masks):
        1. Forward pass through frozen DINOv2
        2. Generate anomaly heatmap via XGBoost
        3. Upsample to create prior
        4. Forward pass through U-Net (with prior injection)
        5. Compute combined loss
        6. Backpropagate (only U-Net parameters)
        7. Update weights
    
    Validate on held-out set
    Save if best validation loss
    Early stop if no improvement for 15 epochs
```

---

## 9. Section 6: Post-Processing

### 9.1 Morphological Operations

```
Input: Raw segmentation mask (H, W)

Step 1: Per-class thresholding
        binary_mask = (probability > 0.5)

Step 2: Morphological erosion (noise removal)
        kernel = ellipse(3, 3)
        eroded = cv2.erode(binary, kernel, iterations=1)

Step 3: Morphological dilation (restore object size)
        dilated = cv2.dilate(eroded, kernel, iterations=2)

Step 4: Connected component analysis
        labels, num_components = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )
```

### 9.2 Instance Extraction

For each connected component:
```
- Bounding box: (x1, y1, x2, y2)
- Pixel mask: binary mask of component
- Area (pixels): count of mask pixels
- Confidence: mean probability within component
- Centroid: (cx, cy)
- Class ID: from segmentation mask
```

### 9.3 Minimum Area Filtering

```
MIN_DEFECT_AREA = 50 pixels

if component.area < MIN_DEFECT_AREA:
    discard as noise
```

---

## 10. Section 7: Area Estimation

### 10.1 Calibration Model

```
Real-world area = Pixel area × (mm_per_pixel)²

Where:
    mm_per_pixel = calibration constant from known reference
    
Example:
    If 10mm reference = 100 pixels
    mm_per_pixel = 10/100 = 0.1 mm/pixel
    mm²_per_pixel² = 0.01 mm²/pixel
    
    For a 500-pixel defect:
    Area = 500 × 0.01 = 5.0 mm²
```

### 10.2 Calibration Procedure

1. Place calibration target with known dimensions
2. Capture image at operational distance
3. Measure reference object in pixels
4. Compute: `mm_per_pixel = known_length_mm / measured_length_pixels`
5. Store in configuration

---

## 11. Output Format

### 11.1 JSON Output Structure

```json
{
    "image_path": "/path/to/image.jpg",
    "timestamp": "2026-01-21T09:00:00",
    "calibration": {
        "mm_per_pixel": 0.1
    },
    "summary": {
        "total_defects": 3,
        "total_area_mm2": 12.5,
        "defects_by_class": {
            "Dust": 1,
            "Scratch": 2
        }
    },
    "defects": [
        {
            "class_id": 3,
            "class_name": "Scratch",
            "confidence": 0.92,
            "area_pixels": 250,
            "area_mm2": 2.5,
            "bounding_box": {
                "x1": 100, "y1": 200, "x2": 350, "y2": 210
            }
        }
    ]
}
```

---

## 12. Experimental Results

### 12.1 XGBoost Patch Classifier Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Background | 0.86 | 0.94 | 0.89 | 202,617 |
| Dust | 0.85 | 0.74 | 0.79 | 92,084 |
| RunDown | 0.84 | 0.61 | 0.71 | 2,305 |
| Scratch | 0.76 | 0.37 → 0.95* | 0.49 → 0.84* | 12,754 |

*After per-class threshold optimization (1% for Scratch)

### 12.2 Key Observations

1. **Background detection** is highly accurate (94% recall)
2. **Dust detection** performs well with standard thresholds
3. **Scratch detection** improved significantly with lower threshold
4. **RunDown** remains challenging due to limited samples (0.7% of patches)

---

## 13. Computational Considerations

### 13.1 Memory Requirements

| Component | GPU Memory |
|-----------|------------|
| DINOv2 (forward) | ~1.5 GB |
| U-Net (training) | ~2.0 GB |
| Batch of 4 images | ~1.0 GB |
| **Total** | **~4.5 GB** |

### 13.2 Inference Time (per image)

| Stage | Time (ms) |
|-------|-----------|
| Preprocessing | ~5 |
| DINOv2 features | ~50 |
| XGBoost inference | ~10 |
| Upsampling | ~2 |
| U-Net inference | ~30 |
| Post-processing | ~5 |
| **Total** | **~100 ms** |

---

## 14. Code Repository Structure

```
spgs_net/
├── config.py                 # All hyperparameters
├── main.py                   # Training/inference pipeline
├── dino/
│   └── feature_extractor.py  # DINOv2 integration
├── ml/
│   └── xgboost_classifier.py # Patch-level classifier
├── anomaly_upsampling/
│   └── upsampler.py          # Prior generation
├── unet/
│   ├── attention_unet.py     # Segmentation network
│   └── losses.py             # Loss functions
├── defect_instance/
│   └── instance_separator.py # Post-processing
├── area_est/
│   └── area_calculator.py    # Physical measurements
└── utils/
    ├── data_utils.py         # Data loading
    └── visualization.py      # Output generation
```

---

## 15. References

1. Oquab, M., et al. "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193 (2023)
2. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI (2015)
3. Chen, L., et al. "XGBoost: A Scalable Tree Boosting System." KDD (2016)
4. Lin, T.Y., et al. "Focal Loss for Dense Object Detection." ICCV (2017)
5. Oktay, O., et al. "Attention U-Net: Learning Where to Look for the Pancreas." MIDL (2018)

---

*Document generated for SPGS-Net implementation - Industrial Multi-Defect Detection System*
