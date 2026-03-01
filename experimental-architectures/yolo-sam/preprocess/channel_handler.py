from typing import Tuple, Union

import cv2
import numpy as np
import torch


def expand_grayscale(
    image: np.ndarray,
    target_channels: int = 3
) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * target_channels, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, target_channels, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == target_channels:
        return image
    else:
        raise ValueError(
            f"Unexpected image shape: {image.shape}. "
            f"Expected (H, W), (H, W, 1), or (H, W, {target_channels})"
        )


def expand_grayscale_tensor(
    tensor: torch.Tensor,
    target_channels: int = 3
) -> torch.Tensor:
    if tensor.ndim == 2:
        # (H, W) -> (C, H, W)
        return tensor.unsqueeze(0).repeat(target_channels, 1, 1)
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        # (1, H, W) -> (C, H, W)
        return tensor.repeat(target_channels, 1, 1)
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        # (B, 1, H, W) -> (B, C, H, W)
        return tensor.repeat(1, target_channels, 1, 1)
    elif tensor.ndim == 3 and tensor.shape[0] == target_channels:
        return tensor
    elif tensor.ndim == 4 and tensor.shape[1] == target_channels:
        return tensor
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image using ImageNet mean and std.
    
    For grayscale images expanded to 3 channels, we use the same
    normalization for consistency with pretrained models.
    
    Args:
        image: Image of shape (H, W, C) with values in [0, 1]
        mean: Channel-wise mean
        std: Channel-wise standard deviation
    
    Returns:
        Normalized image of same shape
    """
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    
    return (image - mean) / std


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Reverse normalization for visualization.
    
    Args:
        image: Normalized image
        mean: Channel-wise mean used for normalization
        std: Channel-wise standard deviation used for normalization
    
    Returns:
        Denormalized image with values in [0, 1]
    """
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    
    image = image * std + mean
    return np.clip(image, 0, 1)


def prepare_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    to_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Complete preprocessing pipeline for model input.
    
    Steps:
    1. Ensure grayscale is expanded to 3 channels
    2. Resize to target size
    3. Normalize (optional)
    4. Convert to tensor (optional)
    
    Args:
        image: Input image (grayscale or RGB)
        target_size: Target (width, height)
        normalize: Whether to apply ImageNet normalization
        to_tensor: Whether to convert to PyTorch tensor
    
    Returns:
        Preprocessed image ready for model input
    """
    # Expand grayscale to 3 channels if needed
    if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
        image = expand_grayscale(image)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Convert to float if needed
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Normalize
    if normalize:
        image = normalize_image(image)
    
    # Convert to tensor
    if to_tensor:
        # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    return image


def prepare_for_sam(
    image: np.ndarray,
    target_size: int = 1024
) -> np.ndarray:
    """
    Prepare image specifically for SAM input.
    
    SAM expects:
    - RGB image (3 channels)
    - Shape: (H, W, 3) with H & W being multiples of 64
    - Values in [0, 255] uint8
    
    Args:
        image: Input grayscale or RGB image
        target_size: Target size (longest edge)
    
    Returns:
        Image ready for SAM encoder
    """
    # Expand grayscale
    if image.ndim == 2:
        image = expand_grayscale(image)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = expand_grayscale(image)
    
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Resize maintaining aspect ratio
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Round to nearest multiple of 64
    new_h = ((new_h + 63) // 64) * 64
    new_w = ((new_w + 63) // 64) * 64
    
    image = cv2.resize(image, (new_w, new_h))
    
    return image


def load_and_preprocess(
    image_path: str,
    target_size: Tuple[int, int] = (640, 640),
    for_yolo: bool = True
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load image from path and preprocess.
    
    Args:
        image_path: Path to image file
        target_size: Target (width, height)
        for_yolo: If True, prepare for YOLO; else prepare for SAM
    
    Returns:
        Tuple of (preprocessed_image, original_size)
    """
    # Load as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_size = image.shape[:2]  # (H, W)
    
    if for_yolo:
        processed = prepare_for_model(image, target_size, normalize=False, to_tensor=False)
    else:
        processed = prepare_for_sam(image, target_size=max(target_size))
    
    return processed, original_size
