"""
SAM Model Loader
Handles loading and caching of Segment Anything Model.
"""
import os
import sys
import torch
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SAM_MODEL_TYPE, SAM_CHECKPOINT_PATH, DEVICE, MODELS_DIR
)

# Global model cache
_sam_model = None
_sam_predictor = None


def download_sam_checkpoint(model_type: str = SAM_MODEL_TYPE) -> Path:
    """
    Download SAM checkpoint if not present.
    
    Model URLs:
    - vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    - vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    - vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    """
    checkpoint_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    
    checkpoint_names = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }
    
    if model_type not in checkpoint_urls:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(checkpoint_urls.keys())}")
    
    checkpoint_path = MODELS_DIR / checkpoint_names[model_type]
    
    if checkpoint_path.exists():
        print(f"✓ SAM checkpoint found: {checkpoint_path}")
        return checkpoint_path
    
    print(f"Downloading SAM {model_type} checkpoint...")
    print(f"URL: {checkpoint_urls[model_type]}")
    print(f"This may take a few minutes...")
    
    import urllib.request
    
    # Create models directory if needed
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rDownloading: {percent:.1f}% ({downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB)", end="")
    
    urllib.request.urlretrieve(
        checkpoint_urls[model_type],
        checkpoint_path,
        reporthook=show_progress
    )
    print(f"\n✓ Downloaded to: {checkpoint_path}")
    
    return checkpoint_path


def load_sam_model(
    model_type: str = SAM_MODEL_TYPE,
    checkpoint_path: Optional[Path] = None,
    device: str = DEVICE
):
    """
    Load SAM model.
    
    Args:
        model_type: One of 'vit_h', 'vit_l', 'vit_b'
        checkpoint_path: Path to checkpoint file (auto-downloads if None)
        device: 'cuda' or 'cpu'
    
    Returns:
        sam: SAM model instance
    """
    global _sam_model
    
    # Return cached model if available
    if _sam_model is not None:
        return _sam_model
    
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise ImportError(
            "segment-anything not installed. Run: pip install segment-anything"
        )
    
    # Download checkpoint if needed
    if checkpoint_path is None:
        checkpoint_path = download_sam_checkpoint(model_type)
    
    if not checkpoint_path.exists():
        checkpoint_path = download_sam_checkpoint(model_type)
    
    print(f"\n{'='*60}")
    print(f"LOADING SAM MODEL: {model_type}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load model
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    
    _sam_model = sam
    print(f"✓ SAM model loaded successfully")
    
    return sam


def get_sam_predictor(
    model_type: str = SAM_MODEL_TYPE,
    checkpoint_path: Optional[Path] = None,
    device: str = DEVICE
):
    """
    Get SAM predictor for interactive segmentation.
    
    The predictor allows providing prompts (points, boxes) for segmentation.
    
    Returns:
        SamPredictor instance
    """
    global _sam_predictor
    
    if _sam_predictor is not None:
        return _sam_predictor
    
    try:
        from segment_anything import SamPredictor
    except ImportError:
        raise ImportError(
            "segment-anything not installed. Run: pip install segment-anything"
        )
    
    sam = load_sam_model(model_type, checkpoint_path, device)
    _sam_predictor = SamPredictor(sam)
    
    print(f"✓ SAM Predictor initialized")
    
    return _sam_predictor


def get_sam_auto_mask_generator(
    model_type: str = SAM_MODEL_TYPE,
    checkpoint_path: Optional[Path] = None,
    device: str = DEVICE,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
):
    """
    Get SAM automatic mask generator.
    
    This generates all masks in an image without prompts.
    
    Returns:
        SamAutomaticMaskGenerator instance
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "segment-anything not installed. Run: pip install segment-anything"
        )
    
    sam = load_sam_model(model_type, checkpoint_path, device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=100,  # Minimum mask area in pixels
    )
    
    print(f"✓ SAM Automatic Mask Generator initialized")
    
    return mask_generator


def clear_sam_cache():
    """Clear cached SAM model to free memory."""
    global _sam_model, _sam_predictor
    
    if _sam_model is not None:
        del _sam_model
        _sam_model = None
    
    if _sam_predictor is not None:
        del _sam_predictor
        _sam_predictor = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✓ SAM cache cleared")


if __name__ == "__main__":
    # Test SAM loading
    print("Testing SAM model loading...")
    
    # This will download the checkpoint if not present
    predictor = get_sam_predictor()
    
    print("\n✓ SAM model loaded and ready for inference!")
