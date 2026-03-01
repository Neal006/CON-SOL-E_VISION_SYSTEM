import os
import sys
import torch
from pathlib import Path
from typing import Optional

try:
    _current_dir = Path(__file__).parent.parent
except NameError:
    _current_dir = Path(os.getcwd())
sys.path.insert(0, str(_current_dir))
from config import DEVICE, MODELS_DIR, MASKRCNN_SCORE_THRESH

_mask_rcnn_model = None


def get_mask_rcnn_model(
    pretrained: bool = True,
    num_classes: int = 91,
    score_thresh: float = MASKRCNN_SCORE_THRESH,
    device: str = DEVICE
):
    """
    Load a pretrained Mask R-CNN model from torchvision.
    
    Args:
        pretrained: Use COCO-pretrained weights
        num_classes: Number of classes (91 for COCO)
        score_thresh: Minimum score for detections
        device: Device to load model on
    
    Returns:
        Mask R-CNN model in eval mode
    """
    global _mask_rcnn_model
    
    if _mask_rcnn_model is not None:
        return _mask_rcnn_model
    
    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    except ImportError:
        raise ImportError("torchvision not installed. Run: pip install torchvision")
    
    print("LOADING MASK R-CNN MODEL")
    print(f"Device: {device}")
    print(f"Score threshold: {score_thresh}")
    
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        model = maskrcnn_resnet50_fpn(weights=weights, box_score_thresh=score_thresh)
    else:
        model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes, box_score_thresh=score_thresh)
    
    model = model.to(device)
    model.eval()
    
    _mask_rcnn_model = model
    print("✓ Mask R-CNN model loaded successfully")
    
    return model


def load_mask_rcnn_model(
    checkpoint_path: Optional[Path] = None,
    num_classes: int = 4,
    score_thresh: float = MASKRCNN_SCORE_THRESH,
    device: str = DEVICE
):
    """
    Load a custom-trained Mask R-CNN model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_classes: Number of classes (including background)
        score_thresh: Minimum score for detections
        device: Device to load model on
    
    Returns:
        Mask R-CNN model in eval mode
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    print(f"LOADING CUSTOM MASK R-CNN MODEL")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num classes: {num_classes}")
    
    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes, box_score_thresh=score_thresh)
    
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded weights from {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    print("✓ Custom Mask R-CNN model loaded")
    return model


def clear_mask_rcnn_cache():
    """Clear the cached Mask R-CNN model to free memory."""
    global _mask_rcnn_model
    if _mask_rcnn_model is not None:
        del _mask_rcnn_model
        _mask_rcnn_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Mask R-CNN cache cleared")


if __name__ == "__main__":
    print("Testing Mask R-CNN model loading...")
    model = get_mask_rcnn_model()
    print(f"Model type: {type(model).__name__}")
    print("✓ Mask R-CNN model loaded and ready for inference!")
