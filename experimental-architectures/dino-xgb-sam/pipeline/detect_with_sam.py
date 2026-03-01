"""
Integrated Defect Detection Pipeline with SAM
Combines DINOv2 + XGBoost detection with SAM pixel-level segmentation.
"""
import os
import sys
import json
import numpy as np
import torch
import joblib
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from scipy import ndimage
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, PROJECT_ROOT,
    PATCH_CLASS_NAMES, PATCH_ID_TO_CLASS, NORMAL_CLASS_ID,
    DINO_MODEL_NAME, DINO_INPUT_SIZE, DINO_PATCH_SIZE,
    DEVICE, CONFIDENCE_THRESHOLD, PIXELS_PER_MM
)

# Defect class names (excluding Normal)
DEFECT_CLASS_NAMES = ["Dust", "RunDown", "Scratch"]
DEFECT_ID_TO_CLASS = {0: "Dust", 1: "RunDown", 2: "Scratch"}
DEFECT_CLASS_TO_ID = {"Dust": 0, "RunDown": 1, "Scratch": 2}

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defect colors (BGR for OpenCV)
DEFECT_COLORS = {
    "Dust": (255, 165, 0),      # Orange
    "RunDown": (0, 165, 255),   # Yellow-orange
    "Scratch": (0, 0, 255),     # Red
}


@dataclass
class DetectedDefect:
    """Represents a detected defect with segmentation."""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    grid_bbox: Tuple[int, int, int, int]  # Original grid coordinates
    patch_count: int
    # SAM segmentation results (filled after SAM processing)
    mask: Optional[np.ndarray] = None
    area_pixels: Optional[int] = None
    area_mm2: Optional[float] = None
    iou_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding mask for JSON serialization)."""
        d = asdict(self)
        d.pop('mask', None)  # Remove mask (not JSON serializable)
        return d


class IntegratedDefectDetector:
    """
    Complete defect detection pipeline:
    1. DINOv2 feature extraction
    2. XGBoost patch classification
    3. Connected component grouping → bounding boxes
    4. SAM pixel-level segmentation
    5. Area estimation
    """
    
    def __init__(
        self,
        model_path: Path = None,
        scaler_path: Path = None,
        use_sam: bool = True,
        device: str = DEVICE
    ):
        self.device = device
        self.use_sam = use_sam
        
        # Model paths
        self.model_path = model_path or MODELS_DIR / "xgb_patch_multiclass.joblib"
        self.scaler_path = scaler_path or MODELS_DIR / "patch_scaler.joblib"
        
        # Check for tuned model
        tuned_model_path = MODELS_DIR / "xgb_tuned.joblib"
        tuned_scaler_path = MODELS_DIR / "scaler_tuned.joblib"
        if tuned_model_path.exists() and tuned_scaler_path.exists():
            print("Using tuned model from hyperparameter optimization")
            self.model_path = tuned_model_path
            self.scaler_path = tuned_scaler_path
        
        # Load DINOv2
        print(f"Loading DINOv2 model: {DINO_MODEL_NAME}...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', DINO_MODEL_NAME)
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()
        
        # Load XGBoost classifier
        print(f"Loading classifier from: {self.model_path}")
        self.classifier = joblib.load(self.model_path)
        
        # Load scaler
        print(f"Loading scaler from: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        
        # Grid size calculation
        self.grid_size = DINO_INPUT_SIZE // DINO_PATCH_SIZE  # 37
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # SAM segmenter (lazy loaded)
        self._sam_segmenter = None
        
        print(f"✓ Detector initialized (grid: {self.grid_size}x{self.grid_size})")
        if use_sam:
            print(f"✓ SAM integration enabled")
    
    def _get_sam_segmenter(self):
        """Lazy load SAM segmenter."""
        if self._sam_segmenter is None and self.use_sam:
            from sam.segment import SAMSegmenter
            self._sam_segmenter = SAMSegmenter(device=self.device)
        return self._sam_segmenter
    
    @torch.no_grad()
    def extract_patch_features(self, image: Image.Image) -> np.ndarray:
        """Extract DINOv2 patch features."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features_dict = self.dino_model.forward_features(img_tensor)
        patch_tokens = features_dict['x_norm_patchtokens']
        
        return patch_tokens.squeeze(0).cpu().numpy()
    
    def classify_patches(
        self,
        patch_features: np.ndarray,
        threshold: float = CONFIDENCE_THRESHOLD
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify each patch using XGBoost."""
        features_scaled = self.scaler.transform(patch_features)
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences, probabilities
    
    def group_defect_regions(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        image_size: Tuple[int, int],  # (width, height)
        threshold: float = CONFIDENCE_THRESHOLD,
        min_patch_count: int = 2
    ) -> List[DetectedDefect]:
        """
        Group high-confidence defect patches into bounding box regions.
        
        Returns:
            List of DetectedDefect objects
        """
        detected_defects = []
        img_w, img_h = image_size
        
        # Scale factors
        scale_x = img_w / self.grid_size
        scale_y = img_h / self.grid_size
        
        # Reshape to grid
        pred_grid = predictions.reshape(self.grid_size, self.grid_size)
        conf_grid = confidences.reshape(self.grid_size, self.grid_size)
        
        # Process each defect class (skip Normal = class 3)
        for class_id, class_name in DEFECT_ID_TO_CLASS.items():
            # Create binary mask for high-confidence patches
            mask = (pred_grid == class_id) & (conf_grid >= threshold)
            
            if not mask.any():
                continue
            
            # Label connected components
            labeled, num_features = ndimage.label(mask)
            
            for region_id in range(1, num_features + 1):
                region_mask = (labeled == region_id)
                patch_count = np.sum(region_mask)
                
                if patch_count < min_patch_count:
                    continue
                
                # Get region bounds (grid coordinates)
                rows, cols = np.where(region_mask)
                gy_min, gy_max = rows.min(), rows.max()
                gx_min, gx_max = cols.min(), cols.max()
                
                # Convert to image coordinates
                x_min = int(gx_min * scale_x)
                y_min = int(gy_min * scale_y)
                x_max = int((gx_max + 1) * scale_x)
                y_max = int((gy_max + 1) * scale_y)
                
                # Average confidence
                avg_confidence = float(conf_grid[region_mask].mean())
                
                defect = DetectedDefect(
                    class_name=class_name,
                    class_id=class_id,
                    confidence=avg_confidence,
                    bbox=(x_min, y_min, x_max, y_max),
                    grid_bbox=(gx_min, gy_min, gx_max, gy_max),
                    patch_count=int(patch_count),
                )
                detected_defects.append(defect)
        
        return detected_defects
    
    def segment_with_sam(
        self,
        image: np.ndarray,
        defects: List[DetectedDefect],
        pixels_per_mm: Optional[float] = PIXELS_PER_MM
    ) -> List[DetectedDefect]:
        """
        Apply SAM segmentation to each detected defect.
        
        Args:
            image: BGR image (H, W, 3)
            defects: List of DetectedDefect objects with bboxes
            pixels_per_mm: Pixels per mm for area conversion
        
        Returns:
            Updated defects with masks and area estimates
        """
        if not self.use_sam or len(defects) == 0:
            return defects
        
        segmenter = self._get_sam_segmenter()
        if segmenter is None:
            print("⚠ SAM not available, skipping segmentation")
            return defects
        
        # Set image for SAM
        segmenter.set_image(image)
        
        for defect in defects:
            try:
                mask, iou, stability = segmenter.segment_box(
                    defect.bbox,
                    multimask_output=True
                )
                
                # Calculate area
                area_pixels = int(np.sum(mask > 0))
                area_mm2 = None
                if pixels_per_mm is not None and pixels_per_mm > 0:
                    area_mm2 = area_pixels / (pixels_per_mm ** 2)
                
                # Update defect
                defect.mask = mask
                defect.area_pixels = area_pixels
                defect.area_mm2 = area_mm2
                defect.iou_score = iou
                
            except Exception as e:
                print(f"⚠ SAM segmentation failed for {defect.class_name}: {e}")
                # Fallback: estimate area from bounding box
                x1, y1, x2, y2 = defect.bbox
                defect.area_pixels = (x2 - x1) * (y2 - y1)
        
        segmenter.reset()
        return defects
    
    def annotate_image(
        self,
        image: np.ndarray,
        defects: List[DetectedDefect],
        show_masks: bool = True,
        show_boxes: bool = True,
        show_area: bool = True,
        mask_alpha: float = 0.4
    ) -> np.ndarray:
        """
        Draw annotations on the image.
        
        Args:
            image: BGR image
            defects: List of DetectedDefect objects
            show_masks: Draw SAM masks
            show_boxes: Draw bounding boxes
            show_area: Show area in labels
            mask_alpha: Mask transparency
        
        Returns:
            Annotated image
        """
        result = image.copy()
        overlay = image.copy()
        
        for defect in defects:
            color = DEFECT_COLORS.get(defect.class_name, (0, 255, 0))
            
            # Draw mask overlay if available
            if show_masks and defect.mask is not None:
                mask_bool = defect.mask > 0
                
                # Resize mask if needed
                if defect.mask.shape[:2] != image.shape[:2]:
                    mask_resized = cv2.resize(
                        defect.mask,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_bool = mask_resized > 0
                
                overlay[mask_bool] = color
            
            if show_boxes:
                x_min, y_min, x_max, y_max = defect.bbox
                cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Prepare label
                if show_area and defect.area_pixels is not None:
                    if defect.area_mm2 is not None:
                        label = f"{defect.class_name} ({defect.confidence:.0%}) {defect.area_mm2:.2f}mm²"
                    else:
                        label = f"{defect.class_name} ({defect.confidence:.0%}) {defect.area_pixels}px"
                else:
                    label = f"{defect.class_name} ({defect.confidence:.0%})"
                
                # Draw label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                label_y = y_min - 5 if y_min > 25 else y_max + text_h + 5
                
                cv2.rectangle(
                    result,
                    (x_min, label_y - text_h - 5),
                    (x_min + text_w + 4, label_y + 5),
                    color, -1
                )
                cv2.putText(
                    result, label,
                    (x_min + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        
        # Blend mask overlay
        if show_masks:
            result = cv2.addWeighted(overlay, mask_alpha, result, 1 - mask_alpha, 0)
        
        return result
    
    def detect(
        self,
        image_path: Union[str, Path],
        threshold: float = CONFIDENCE_THRESHOLD,
        apply_sam: bool = True,
        pixels_per_mm: Optional[float] = PIXELS_PER_MM
    ) -> Tuple[np.ndarray, List[DetectedDefect], Dict]:
        """
        Run complete detection pipeline.
        
        Args:
            image_path: Path to input image
            threshold: Confidence threshold
            apply_sam: Apply SAM segmentation
            pixels_per_mm: Pixels per mm for area calibration
        
        Returns:
            annotated_image: Annotated image with boxes and masks
            defects: List of detected defects
            metadata: Detection metadata
        """
        image_path = Path(image_path)
        
        # Load image
        image_pil = Image.open(image_path)
        image_np = np.array(image_pil.convert('RGB'))
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        img_w, img_h = image_pil.size
        
        print(f"\n{'='*60}")
        print(f"DETECTING DEFECTS: {image_path.name}")
        print(f"Image size: {img_w}x{img_h}")
        print(f"{'='*60}")
        
        # Step 1: Extract DINOv2 patch features
        print("1. Extracting DINOv2 patch features...")
        patch_features = self.extract_patch_features(image_pil)
        print(f"   → {patch_features.shape[0]} patches extracted")
        
        # Step 2: Classify patches with XGBoost
        print("2. Classifying patches...")
        predictions, confidences, probabilities = self.classify_patches(
            patch_features, threshold
        )
        
        # Step 3: Group into defect regions
        print("3. Grouping defect regions...")
        defects = self.group_defect_regions(
            predictions, confidences,
            image_size=(img_w, img_h),
            threshold=threshold
        )
        print(f"   → {len(defects)} defect region(s) detected")
        
        for defect in defects:
            print(f"      - {defect.class_name}: {defect.confidence:.1%} confidence")
        
        # Step 4: SAM segmentation (if enabled)
        if apply_sam and self.use_sam and len(defects) > 0:
            print("4. Applying SAM segmentation...")
            defects = self.segment_with_sam(image_bgr, defects, pixels_per_mm)
            print(f"   → Pixel-level masks generated")
            
            for defect in defects:
                if defect.area_pixels:
                    area_str = f"{defect.area_mm2:.2f}mm²" if defect.area_mm2 else f"{defect.area_pixels}px"
                    print(f"      - {defect.class_name}: area = {area_str}")
        
        # Step 5: Annotate image
        print("5. Annotating image...")
        annotated = self.annotate_image(
            image_bgr, defects,
            show_masks=(apply_sam and self.use_sam),
            show_boxes=True,
            show_area=True
        )
        
        # Prepare metadata
        metadata = {
            'image_path': str(image_path),
            'image_size': {'width': img_w, 'height': img_h},
            'threshold': threshold,
            'pixels_per_mm': pixels_per_mm,
            'sam_applied': apply_sam and self.use_sam,
            'num_defects': len(defects),
            'defects': [d.to_dict() for d in defects],
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"{'='*60}")
        print(f"✓ Detection complete: {len(defects)} defect(s) found")
        
        return annotated, defects, metadata
    
    def detect_and_save(
        self,
        image_path: Union[str, Path],
        output_dir: Path = None,
        threshold: float = CONFIDENCE_THRESHOLD,
        apply_sam: bool = True,
        pixels_per_mm: Optional[float] = PIXELS_PER_MM
    ) -> Dict:
        """
        Run detection and save results.
        
        Returns:
            metadata: Detection metadata including output paths
        """
        output_dir = output_dir or OUTPUT_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run detection
        annotated, defects, metadata = self.detect(
            image_path, threshold, apply_sam, pixels_per_mm
        )
        
        # Save annotated image
        input_name = Path(image_path).stem
        annotated_path = output_dir / f"{input_name}_detected.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        metadata['annotated_image_path'] = str(annotated_path)
        
        # Save metadata as JSON
        metadata_path = output_dir / f"{input_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        metadata['metadata_path'] = str(metadata_path)
        
        # Save individual masks if SAM was applied
        if apply_sam and self.use_sam:
            masks_dir = output_dir / f"{input_name}_masks"
            masks_dir.mkdir(exist_ok=True)
            
            for i, defect in enumerate(defects):
                if defect.mask is not None:
                    mask_path = masks_dir / f"mask_{i}_{defect.class_name}.png"
                    cv2.imwrite(str(mask_path), defect.mask)
            
            metadata['masks_dir'] = str(masks_dir)
        
        print(f"\n✓ Results saved to: {output_dir}")
        print(f"  - Annotated image: {annotated_path.name}")
        print(f"  - Metadata: {metadata_path.name}")
        
        return metadata


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Defect Detection with SAM')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold')
    parser.add_argument('--no-sam', action='store_true', help='Disable SAM segmentation')
    parser.add_argument('--pixels-per-mm', type=float, default=None,
                       help='Pixels per mm for area calibration')
    
    args = parser.parse_args()
    
    if args.image is None:
        # Interactive mode
        print("\n" + "="*60)
        print("INTEGRATED DEFECT DETECTION PIPELINE")
        print("DINOv2 + XGBoost + SAM")
        print("="*60)
        
        image_path = input("\nEnter image path: ").strip().strip('"\'')
        
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            return
    else:
        image_path = args.image
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = IntegratedDefectDetector(use_sam=not args.no_sam)
    
    # Run detection
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    metadata = detector.detect_and_save(
        image_path,
        output_dir=output_dir,
        threshold=args.threshold,
        apply_sam=not args.no_sam,
        pixels_per_mm=args.pixels_per_mm
    )
    
    print("\n" + "="*60)
    print("DETECTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
