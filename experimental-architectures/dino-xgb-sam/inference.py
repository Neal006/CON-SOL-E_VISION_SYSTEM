import os
import sys
import numpy as np
import torch
import joblib
from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS_DIR, FEATURES_DIR,
    CLASS_NAMES, ID_TO_CLASS, NUM_CLASSES,
    DINO_MODEL_NAME, DINO_INPUT_SIZE, DEVICE
)


class MultiClassAnomalyDetector:
    """Multi-class anomaly detector for Scratch, Dust, and RunDown defects."""
    
    def __init__(self, model_path: Path = None, scaler_path: Path = None):
        
        self.model_path = model_path or MODELS_DIR / "xgb_multiclass.joblib"
        self.scaler_path = scaler_path or MODELS_DIR / "scaler.joblib"
        
        print(f"Loading DINOv2 model: {DINO_MODEL_NAME}...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', DINO_MODEL_NAME)
        self.dino_model = self.dino_model.to(DEVICE)
        self.dino_model.eval()
        
        print(f"Loading classifier from: {self.model_path}")
        self.classifier = joblib.load(self.model_path)
        
        print(f"Loading scaler from: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✓ Detector initialized successfully")
    
    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        features = self.dino_model(img_tensor)
        
        return features.cpu().numpy()
    
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        
        features = self.extract_features(image)
        
        features_scaled = self.scaler.transform(features)
        
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        predicted_class = CLASS_NAMES[prediction]
        confidence = probabilities[prediction]
        
        class_probs = {CLASS_NAMES[i]: prob for i, prob in enumerate(probabilities)}
        
        return predicted_class, confidence, class_probs
    
    def predict_batch(self, images: List[Image.Image]) -> List[Tuple[str, float, Dict[str, float]]]:
        
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def predict_from_path(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        image = Image.open(image_path)
        return self.predict(image)


def demo():
    print("MULTI-CLASS ANOMALY DETECTOR DEMO")
    
    detector = MultiClassAnomalyDetector()
    
    from config import TRAIN_IMAGES_PATH
    sample_images = list(TRAIN_IMAGES_PATH.glob("*.jpg"))[:3]
    
    print(f"\nRunning predictions on {len(sample_images)} sample images...")
    for img_path in sample_images:
        predicted_class, confidence, probs = detector.predict_from_path(str(img_path))
        
        print(f"\n  Image: {img_path.name}")
        print(f"  Predicted: {predicted_class} (confidence: {confidence:.2%})")
        print(f"  Probabilities:")
        for class_name, prob in probs.items():
            print(f"    - {class_name}: {prob:.2%}")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo()
