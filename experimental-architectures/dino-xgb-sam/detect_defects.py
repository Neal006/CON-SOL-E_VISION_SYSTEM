"""
Defect Detection and Annotation Script
Detects Scratch, Dust, and RunDown defects using DINOv2 patch-level classification.
Outputs annotated images with bounding boxes around defect regions.
"""
import os
import sys
import numpy as np
import torch
import joblib
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Tuple
from scipy import ndimage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS_DIR, PROJECT_ROOT,
    PATCH_CLASS_NAMES, PATCH_ID_TO_CLASS, NORMAL_CLASS_ID,
    DINO_MODEL_NAME, DINO_INPUT_SIZE, DINO_PATCH_SIZE,
    DEVICE, CONFIDENCE_THRESHOLD
)

# Defect class names (excluding Normal)
DEFECT_CLASS_NAMES = ["Dust", "RunDown", "Scratch"]
DEFECT_ID_TO_CLASS = {0: "Dust", 1: "RunDown", 2: "Scratch"}

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defect colors (BGR for OpenCV)
DEFECT_COLORS = {
    "Dust": (255, 165, 0),      # Orange
    "RunDown": (0, 165, 255),   # Orange-yellow
    "Scratch": (0, 0, 255),     # Red
}


class DefectDetector:
    """Patch-level defect detector using DINOv2 and XGBoost."""
    
    def __init__(self, model_path: Path = None, scaler_path: Path = None):
        # Use patch-level model (trained with Normal class)
        self.model_path = model_path or MODELS_DIR / "xgb_patch_multiclass.joblib"
        self.scaler_path = scaler_path or MODELS_DIR / "patch_scaler.joblib"
        
        print(f"Loading DINOv2 model: {DINO_MODEL_NAME}...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', DINO_MODEL_NAME)
        self.dino_model = self.dino_model.to(DEVICE)
        self.dino_model.eval()
        
        print(f"Loading classifier from: {self.model_path}")
        self.classifier = joblib.load(self.model_path)
        
        print(f"Loading scaler from: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        
        # Calculate grid size (37 for 518x518 with 14px patches)
        self.grid_size = DINO_INPUT_SIZE // DINO_PATCH_SIZE
        
        self.transform = transforms.Compose([
            transforms.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✓ Detector initialized (grid: {self.grid_size}x{self.grid_size})")
    
    @torch.no_grad()
    def extract_patch_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract features for all patches using DINOv2.
        
        Returns:
            np.ndarray: Shape (grid_size * grid_size, embed_dim) - features for each patch
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        # Get all output features (including patch tokens)
        # DINOv2 forward_features returns dict with 'x_norm_patchtokens'
        features_dict = self.dino_model.forward_features(img_tensor)
        
        # Get patch tokens (excluding CLS token)
        # Shape: (1, num_patches, embed_dim)
        patch_tokens = features_dict['x_norm_patchtokens']
        
        return patch_tokens.squeeze(0).cpu().numpy()
    
    def classify_patches(
        self, 
        patch_features: np.ndarray, 
        threshold: float = CONFIDENCE_THRESHOLD
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify each patch and return predictions.
        
        Returns:
            predictions: (num_patches,) array of class IDs
            confidences: (num_patches,) array of confidence scores
            probabilities: (num_patches, num_classes) array of probabilities
        """
        # Scale features
        features_scaled = self.scaler.transform(patch_features)
        
        # Get predictions and probabilities
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        # Get confidence for each prediction
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences, probabilities
    
    def group_defect_regions(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        threshold: float = CONFIDENCE_THRESHOLD,
        min_patch_count: int = 2
    ) -> Dict[str, List[Dict]]:
        """
        Group high-confidence defect patches into bounding box regions.
        Filters out patches classified as Normal.
        
        Returns:
            Dict mapping defect type to list of bounding boxes
        """
        defect_regions = {class_name: [] for class_name in DEFECT_CLASS_NAMES}
        
        # Reshape to grid
        pred_grid = predictions.reshape(self.grid_size, self.grid_size)
        conf_grid = confidences.reshape(self.grid_size, self.grid_size)
        
        # Process each DEFECT class (skip Normal = class 3)
        for class_id, class_name in DEFECT_ID_TO_CLASS.items():
            # Create binary mask for high-confidence patches of this class
            mask = (pred_grid == class_id) & (conf_grid >= threshold)
            
            if not mask.any():
                continue
            
            # Label connected components
            labeled, num_features = ndimage.label(mask)
            
            # Get bounding box for each component
            for region_id in range(1, num_features + 1):
                region_mask = (labeled == region_id)
                patch_count = np.sum(region_mask)
                
                # Skip small regions
                if patch_count < min_patch_count:
                    continue
                
                # Get region bounds (in grid coordinates)
                rows, cols = np.where(region_mask)
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                
                # Calculate average confidence for this region
                avg_confidence = conf_grid[region_mask].mean()
                
                defect_regions[class_name].append({
                    'grid_bbox': (x_min, y_min, x_max, y_max),
                    'confidence': float(avg_confidence),
                    'patch_count': int(patch_count)
                })
        
        return defect_regions
    
    def annotate_image(
        self,
        image: np.ndarray,
        defect_regions: Dict[str, List[Dict]],
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        """
        annotated = image.copy()
        h, w = image.shape[:2]
        
        # Calculate scale factors from grid to image coordinates
        scale_x = w / self.grid_size
        scale_y = h / self.grid_size
        
        for class_name, regions in defect_regions.items():
            color = DEFECT_COLORS.get(class_name, (0, 255, 0))
            
            for region in regions:
                gx_min, gy_min, gx_max, gy_max = region['grid_bbox']
                
                # Convert grid coordinates to image coordinates
                x_min = int(gx_min * scale_x)
                y_min = int(gy_min * scale_y)
                x_max = int((gx_max + 1) * scale_x)
                y_max = int((gy_max + 1) * scale_y)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), 
                            color, line_thickness)
                
                # Prepare label
                label = f"{class_name} ({region['confidence']:.0%})"
                
                # Draw label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                
                # Position label above bbox, or below if too close to top
                label_y = y_min - 5 if y_min > 25 else y_max + text_h + 5
                
                cv2.rectangle(annotated, 
                            (x_min, label_y - text_h - 5),
                            (x_min + text_w + 4, label_y + 5),
                            color, -1)
                
                cv2.putText(annotated, label, 
                          (x_min + 2, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated
    
    def detect_and_annotate(
        self,
        image_path: str,
        threshold: float = CONFIDENCE_THRESHOLD
    ) -> Tuple[np.ndarray, Dict[str, List[Dict]]]:
        """
        Main detection pipeline.
        
        Args:
            image_path: Path to input image
            threshold: Confidence threshold for detection
            
        Returns:
            annotated_image: Image with bounding boxes
            defect_regions: Dict of detected defect regions
        """
        # Load image
        image = Image.open(image_path)
        image_np = np.array(image.convert('RGB'))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.size}")
        
        # Extract patch features
        print("Extracting patch features...")
        patch_features = self.extract_patch_features(image)
        print(f"Extracted {patch_features.shape[0]} patch features")
        
        # Classify patches
        print("Classifying patches...")
        predictions, confidences, _ = self.classify_patches(patch_features, threshold)
        
        # Group into regions
        print("Grouping defect regions...")
        defect_regions = self.group_defect_regions(
            predictions, confidences, threshold
        )
        
        # Count detections
        total_defects = sum(len(regions) for regions in defect_regions.values())
        print(f"Found {total_defects} defect region(s)")
        
        for class_name, regions in defect_regions.items():
            if regions:
                print(f"  - {class_name}: {len(regions)} region(s)")
        
        # Annotate image
        annotated = self.annotate_image(image_np, defect_regions)
        
        return annotated, defect_regions


def main():
    print("DEFECT DETECTION AND ANNOTATION")
    print("\nDefect types: Scratch, Dust, RunDown")
    print(f"Output folder: {OUTPUT_DIR}")
    print()
    
    image_path = input("Enter image path: ").strip()
    
    image_path = image_path.strip('"\'')
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    print("\nInitializing detector...")
    detector = DefectDetector()
    
    print("\n" + "-" * 40)
    annotated_image, defect_regions = detector.detect_and_annotate(image_path)
    
    input_name = Path(image_path).stem
    output_path = OUTPUT_DIR / f"{input_name}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated_image)
    
    print("\n" + "=" * 60)
    print(f"✓ Annotated image saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
