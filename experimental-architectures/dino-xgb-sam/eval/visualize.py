"""
Visualization Module for Multi-Class Anomaly Detection
Generates prediction images with ground truth vs prediction overlays
"""
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, MASKS_DIR,
    EVALUATION_DIR, PREDICTIONS_DIR,
    CLASS_NAMES, ID_TO_CLASS, NUM_CLASSES
)


# Define colors for each class (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 255),    # Dust - Yellow
    1: (0, 165, 255),    # RunDown - Orange
    2: (0, 0, 255),      # Scratch - Red
}

CLASS_COLORS_RGB = {
    0: (255, 255, 0),    # Dust - Yellow
    1: (255, 165, 0),    # RunDown - Orange
    2: (255, 0, 0),      # Scratch - Red
}


def load_predictions():
    """Load predictions from JSON file."""
    predictions_path = EVALUATION_DIR / "predictions.json"
    with open(predictions_path, 'r') as f:
        return json.load(f)


def parse_yolov8_label(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """Parse YOLOv8 segmentation label file."""
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    polygon.append([x, y])
            
            if len(polygon) >= 3:
                annotations.append({
                    'class_id': class_id,
                    'polygon': np.array(polygon, dtype=np.int32)
                })
    
    return annotations


def draw_annotations_on_image(image: np.ndarray, annotations: List[Dict], alpha: float = 0.4) -> np.ndarray:
    """Draw polygon annotations on image with transparency."""
    overlay = image.copy()
    
    for ann in annotations:
        class_id = ann['class_id']
        polygon = ann['polygon']
        color = CLASS_COLORS.get(class_id, (128, 128, 128))
        
        # Fill polygon with transparency
        cv2.fillPoly(overlay, [polygon], color)
        
        # Draw polygon outline
        cv2.polylines(image, [polygon], True, color, 2)
    
    # Blend overlay with original
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result


def create_prediction_visualization(
    image_path: str,
    actual_class: int,
    predicted_class: int,
    output_path: Path
):
    """Create visualization comparing actual vs predicted with ground truth overlay."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return
    
    img_height, img_width = img.shape[:2]
    
    # Get corresponding label file
    stem = Path(image_path).stem
    label_path = TRAIN_LABELS_PATH / f"{stem}.txt"
    
    # Parse annotations
    annotations = parse_yolov8_label(label_path, img_width, img_height)
    
    # Create visualization with ground truth overlay
    vis_img = draw_annotations_on_image(img.copy(), annotations)
    
    # Add text labels
    actual_name = CLASS_NAMES[actual_class]
    predicted_name = CLASS_NAMES[predicted_class]
    is_correct = actual_class == predicted_class
    
    # Create info bar at top
    bar_height = 60
    vis_with_bar = np.zeros((img_height + bar_height, img_width, 3), dtype=np.uint8)
    vis_with_bar[bar_height:, :] = vis_img
    
    # Background color for bar
    bar_color = (0, 100, 0) if is_correct else (0, 0, 100)  # Green if correct, red if wrong
    vis_with_bar[:bar_height, :] = bar_color
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    actual_text = f"Actual: {actual_name}"
    pred_text = f"Predicted: {predicted_name}"
    status_text = "CORRECT" if is_correct else "WRONG"
    
    cv2.putText(vis_with_bar, actual_text, (10, 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(vis_with_bar, pred_text, (10, 50), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(vis_with_bar, status_text, (img_width - 120, 40), font, font_scale, (255, 255, 255), thickness)
    
    # Draw legend
    legend_x = img_width - 200
    for i, name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[i]
        cv2.rectangle(vis_with_bar, (legend_x, 10 + i*18), (legend_x + 15, 22 + i*18), color, -1)
        cv2.putText(vis_with_bar, name, (legend_x + 20, 22 + i*18), font, 0.4, (255, 255, 255), 1)
    
    # Save
    cv2.imwrite(str(output_path), vis_with_bar)


def generate_sample_predictions(num_samples: int = 30):
    """Generate prediction visualization images for sample of test set."""
    print(f"\n{'='*60}")
    print("GENERATING PREDICTION VISUALIZATION IMAGES")
    print(f"{'='*60}")
    
    predictions = load_predictions()
    test_data = predictions['test']
    
    y_true = test_data['y_true']
    y_pred = test_data['y_pred']
    paths = test_data['paths']
    
    # Create output directories
    correct_dir = PREDICTIONS_DIR / "correct"
    incorrect_dir = PREDICTIONS_DIR / "incorrect"
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)
    
    # Separate correct and incorrect predictions
    correct_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    incorrect_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    
    print(f"Test set: {len(correct_indices)} correct, {len(incorrect_indices)} incorrect")
    
    # Sample from each
    np.random.seed(42)
    num_correct = min(num_samples // 2, len(correct_indices))
    num_incorrect = min(num_samples // 2, len(incorrect_indices))
    
    correct_sample = np.random.choice(correct_indices, num_correct, replace=False) if correct_indices else []
    incorrect_sample = np.random.choice(incorrect_indices, num_incorrect, replace=False) if incorrect_indices else []
    
    # Generate visualizations for correct predictions
    print(f"\nGenerating {len(correct_sample)} correct prediction visualizations...")
    for i, idx in enumerate(correct_sample):
        output_path = correct_dir / f"correct_{i+1:03d}_{CLASS_NAMES[y_true[idx]]}.png"
        create_prediction_visualization(paths[idx], y_true[idx], y_pred[idx], output_path)
    print(f"✓ Saved to: {correct_dir}")
    
    # Generate visualizations for incorrect predictions
    print(f"\nGenerating {len(incorrect_sample)} incorrect prediction visualizations...")
    for i, idx in enumerate(incorrect_sample):
        actual = CLASS_NAMES[y_true[idx]]
        predicted = CLASS_NAMES[y_pred[idx]]
        output_path = incorrect_dir / f"wrong_{i+1:03d}_{actual}_as_{predicted}.png"
        create_prediction_visualization(paths[idx], y_true[idx], y_pred[idx], output_path)
    print(f"✓ Saved to: {incorrect_dir}")
    
    # Generate all incorrect predictions (for detailed analysis)
    all_incorrect_dir = PREDICTIONS_DIR / "all_incorrect"
    all_incorrect_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating ALL {len(incorrect_indices)} incorrect prediction visualizations...")
    for i, idx in enumerate(incorrect_indices):
        actual = CLASS_NAMES[y_true[idx]]
        predicted = CLASS_NAMES[y_pred[idx]]
        output_path = all_incorrect_dir / f"wrong_{i+1:03d}_{actual}_as_{predicted}.png"
        create_prediction_visualization(paths[idx], y_true[idx], y_pred[idx], output_path)
    print(f"✓ Saved to: {all_incorrect_dir}")
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput locations:")
    print(f"  - Correct predictions:   {correct_dir}")
    print(f"  - Sample incorrect:      {incorrect_dir}")
    print(f"  - All incorrect:         {all_incorrect_dir}")


def generate_class_distribution_plot():
    """Generate class distribution comparison plot."""
    predictions = load_predictions()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, split in enumerate(['train', 'val', 'test']):
        y_true = np.array(predictions[split]['y_true'])
        y_pred = np.array(predictions[split]['y_pred'])
        
        # Count actual and predicted
        actual_counts = [np.sum(y_true == i) for i in range(NUM_CLASSES)]
        pred_counts = [np.sum(y_pred == i) for i in range(NUM_CLASSES)]
        
        x = np.arange(NUM_CLASSES)
        width = 0.35
        
        axes[idx].bar(x - width/2, actual_counts, width, label='Actual', color='steelblue')
        axes[idx].bar(x + width/2, pred_counts, width, label='Predicted', color='coral')
        
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'{split.capitalize()} Set')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        axes[idx].legend()
        
        # Add count labels
        for i, (a, p) in enumerate(zip(actual_counts, pred_counts)):
            axes[idx].text(i - width/2, a + 2, str(a), ha='center', fontsize=8)
            axes[idx].text(i + width/2, p + 2, str(p), ha='center', fontsize=8)
    
    plt.tight_layout()
    plot_path = EVALUATION_DIR / "class_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class distribution plot to: {plot_path}")


if __name__ == "__main__":
    generate_sample_predictions(num_samples=30)
    generate_class_distribution_plot()
