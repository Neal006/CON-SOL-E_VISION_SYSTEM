import os
import sys
from pathlib import Path
import subprocess

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                "torch>=2.6.0", "torchvision>=0.16.0", "timm>=0.9.12", 
                "opencv-python>=4.8.0", "albumentations>=1.3.1", "pillow>=10.0.0", 
                "numpy>=1.24.0", "matplotlib>=3.8.0", "seaborn>=0.13.0", 
                "pyyaml>=6.0", "tqdm>=4.66.0", "scikit-learn>=1.3.0", 
                "tensorboard>=2.15.0"], check=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
import logging
from typing import Optional, Dict, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

working_dir = Path("/kaggle/working")
os.chdir(working_dir)

(working_dir / "configs").mkdir(exist_ok=True)
(working_dir / "checkpoints").mkdir(exist_ok=True)
(working_dir / "logs").mkdir(exist_ok=True)

config_yaml = """model:
  encoder: "dinov2_vitb14"
  encoder_frozen: true  # START FROZEN
  skip_layers: [3, 7, 11]
  decoder_channels: [256, 128, 64]
  num_classes: 5

training:
  batch_size: 2
  num_epochs: 60
  learning_rate: 0.0002  # Effective: 5e-5 (after scaling) - 40x MORE POWER
  weight_decay: 0.01  # Keep high for regularization
  warmup_epochs: 5
  unfreeze_at_epoch: 15
  gradient_clip: 1.0
  accumulation_steps: 4
  reference_batch_size: 8

loss:
  dice_weight: 1.0   # Doubled importance
  focal_weight: 1.0  # Doubled importance
  focal_alpha: 1.0
  focal_gamma: 2.0
  # [Background, Chipping, Dust, Rundown, Scratch]
  # Downweight background (0.1) significantly vs defects (1.0)
  class_weights: [0.1, 1.0, 1.0, 1.0, 1.0]
    
data:
  image_size: 476   # Reduced from 518 (20% fewer pixels -> Faster)
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  num_workers: 2    # Restored to 2 for optimal data loading on Kaggle

augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5    # Increased from 0.3
  rotation_limit: 20    # Increased from 15
  brightness_limit: 0.3 # Increased from 0.2
  contrast_limit: 0.3   # Increased from 0.2
  gaussian_blur_p: 0.1  # REDUCED (blur hurts scratch detection!)
  clahe: 0.3
  elastic_transform_p: 0.5 # Increased from 0.3 (great for organic defects)

paths:
  data_root: "/kaggle/input/whitedoor"
  save_dir: "./checkpoints"
  log_dir: "./logs"
"""

with open("configs/config.yaml", "w") as f:
    f.write(config_yaml)

config = yaml.safe_load(config_yaml)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred_prob = F.softmax(pred, dim=1)
        
        valid_mask = (target != self.ignore_index)
        target_masked = target.clone().long()
        target_masked[~valid_mask] = 0
        
        target_one_hot = F.one_hot(target_masked, num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        valid_mask = valid_mask.unsqueeze(1).expand_as(pred_prob)
        pred_prob = pred_prob * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        dims = (0, 2, 3)
        intersection = (pred_prob * target_one_hot).sum(dim=dims)
        pred_sum = pred_prob.sum(dim=dims)
        target_sum = target_one_hot.sum(dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Exclude background (class 0) from average Dice computation
        # This prevents the dominant background class (Dice ~0.99) from masking poor defect performance
        return 1.0 - dice[1:].mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 class_weights: Optional[torch.Tensor] = None, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate CE loss without weights first to get true probabilities (pt)
        ce_loss = F.cross_entropy(pred, target, reduction='none', 
                                  ignore_index=self.ignore_index)
        
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights manually if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            # Create weight map matching target shape
            # Handle ignore_index by temporarily mapping it to 0 (loss is masked anyway)
            target_safe = target.clone()
            target_safe[target == self.ignore_index] = 0
            sample_weights = weights[target_safe]
            focal_loss = focal_loss * sample_weights
        
        valid_mask = (target != self.ignore_index)
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None, ignore_index: int = -100):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, 
                                    class_weights=class_weights, ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        loss = 0.0
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        return loss


class SegmentationMetrics:
    """Calculate comprehensive segmentation metrics including Recall and Detection Accuracy."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None, 
                 ignore_index: int = -100):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with new predictions and targets.
        
        Args:
            pred: Predicted class indices of shape (B, H, W)
            target: Ground truth of shape (B, H, W)
        """
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        valid_mask = (target != self.ignore_index)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += ((target == i) & (pred == j)).sum()
        
        self.total_samples += len(pred)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics from confusion matrix.
        
        Returns:
            Dictionary containing all metrics including Recall and Detection Accuracy
        """
        cm = self.confusion_matrix
        metrics = {}
        
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        
        epsilon = 1e-7
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        iou = tp / (tp + fp + fn + epsilon)
        
        pixel_accuracy = tp.sum() / (cm.sum() + epsilon)
        
        detection_accuracy = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            total_class_pixels = cm[i, :].sum()
            if total_class_pixels > 0:
                detection_accuracy[i] = tp[i] / total_class_pixels
            else:
                detection_accuracy[i] = 0.0
        
        metrics['pixel_accuracy'] = float(pixel_accuracy)
        metrics['mean_iou'] = float(np.mean(iou))
        metrics['mean_dice'] = float(np.mean(f1_score))
        metrics['mean_precision'] = float(np.mean(precision))
        metrics['mean_recall'] = float(np.mean(recall))
        metrics['mean_f1_score'] = float(np.mean(f1_score))
        metrics['mean_detection_accuracy'] = float(np.mean(detection_accuracy))
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'iou_{class_name}'] = float(iou[i])
            metrics[f'dice_{class_name}'] = float(f1_score[i])
            metrics[f'precision_{class_name}'] = float(precision[i])
            metrics[f'recall_{class_name}'] = float(recall[i])
            metrics[f'f1_score_{class_name}'] = float(f1_score[i])
            metrics[f'detection_accuracy_{class_name}'] = float(detection_accuracy[i])
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print("\n" + "="*70)
        print("EVALUATION METRICS")
        print("="*70)
        print(f"{'Metric':<30} {'Value':>10}")
        print("-"*70)
        print(f"{'Pixel Accuracy':<30} {metrics['pixel_accuracy']:>10.4f}")
        print(f"{'Mean IoU':<30} {metrics['mean_iou']:>10.4f}")
        print(f"{'Mean Dice':<30} {metrics['mean_dice']:>10.4f}")
        print(f"{'Mean Precision':<30} {metrics['mean_precision']:>10.4f}")
        print(f"{'Mean Recall':<30} {metrics['mean_recall']:>10.4f}")
        print(f"{'Mean F1 Score':<30} {metrics['mean_f1_score']:>10.4f}")
        print(f"{'Mean Detection Accuracy':<30} {metrics['mean_detection_accuracy']:>10.4f}")
        print("-"*70)
        print("\nPer-Class Metrics:")
        print("-"*70)
        for class_name in self.class_names:
            print(f"\n{class_name.upper()}:")
            print(f"  IoU:                {metrics[f'iou_{class_name}']:>10.4f}")
            print(f"  Dice:               {metrics[f'dice_{class_name}']:>10.4f}")
            print(f"  Precision:          {metrics[f'precision_{class_name}']:>10.4f}")
            print(f"  Recall:             {metrics[f'recall_{class_name}']:>10.4f}")
            print(f"  F1 Score:           {metrics[f'f1_score_{class_name}']:>10.4f}")
            print(f"  Detection Accuracy: {metrics[f'detection_accuracy_{class_name}']:>10.4f}")
        print("="*70)


def get_train_transform(config: dict) -> A.Compose:
    aug_cfg = config.get('augmentation', {})
    data_cfg = config.get('data', {})
    image_size = data_cfg.get('image_size', 518)
    
    return A.Compose([
        A.HorizontalFlip(p=aug_cfg.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=aug_cfg.get('vertical_flip', 0.3)),
        A.Rotate(limit=aug_cfg.get('rotation_limit', 15), border_mode=0, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get('brightness_limit', 0.2),
            contrast_limit=aug_cfg.get('contrast_limit', 0.2), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=aug_cfg.get('gaussian_blur_p', 0.3)),
        A.CoarseDropout(num_holes_range=(1, 8),
                       hole_height_range=(image_size // 40, image_size // 20),
                       hole_width_range=(image_size // 40, image_size // 20),
                       fill=0, p=0.3),
        A.ElasticTransform(p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transform(config: dict) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, label_dir: str, image_size: int = 518,
                 transform=None, num_classes: int = 2, is_training: bool = True,
                 image_files: Optional[List] = None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.transform = transform
        self.num_classes = num_classes
        self.is_training = is_training
        
        if image_files is not None:
            self.image_files = image_files
        else:
            self.image_files = sorted(list(self.image_dir.glob("*.jpg")) + 
                                     list(self.image_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = self.label_dir / img_path.name
        if label_path.exists():
            mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), 
                         interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {'image': image, 'mask': mask.long(), 'image_path': str(img_path)}


def split_dataset(image_dir: str, train_ratio: float = 0.6, val_ratio: float = 0.2, 
                  test_ratio: float = 0.2, seed: int = 42) -> Tuple[List, List, List]:
    """Split dataset into train/val/test sets.
    
    Args:
        image_dir: Directory containing images
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    image_dir = Path(image_dir)
    all_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    
    if len(all_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    np.random.seed(seed)
    indices = np.random.permutation(len(all_files))
    
    train_size = int(len(all_files) * train_ratio)
    val_size = int(len(all_files) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]
    test_files = [all_files[i] for i in test_indices]
    
    logger.info(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    return train_files, val_files, test_files


class DecoderBlock(nn.Module):
    """Decoder block with GroupNorm instead of BatchNorm.
    
    GroupNorm computes normalization statistics per-sample (not across the batch),
    so it is immune to small-batch gradient noise. This is critical when training
    with batch_size=1-2 and gradient accumulation.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 num_groups: int = 8):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                          kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 
                              kernel_size=3, padding=1)
        # GroupNorm: num_groups divides out_channels evenly
        gn1_groups = min(num_groups, out_channels)
        while out_channels % gn1_groups != 0:
            gn1_groups -= 1
        self.gn1 = nn.GroupNorm(gn1_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        gn2_groups = gn1_groups  # Same channel count, same groups
        self.gn2 = nn.GroupNorm(gn2_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Resize skip connection if spatial dimensions don't match
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class SegmentationModel(nn.Module):
    def __init__(self, encoder_name: str = "dinov2_vitb14", num_classes: int = 3,
                 decoder_channels: List[int] = [256, 128, 64],
                 skip_layers: List[int] = [3, 7, 11], encoder_frozen: bool = False):
        super().__init__()
        
        self.encoder = torch.hub.load('facebookresearch/dinov2', encoder_name)
        self.encoder_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        self.skip_layers = skip_layers
        
        if encoder_frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # GroupNorm in bottleneck (immune to small-batch statistics noise)
        bottleneck_ch = decoder_channels[0] * 2
        gn_groups_bn = min(8, bottleneck_ch)
        while bottleneck_ch % gn_groups_bn != 0:
            gn_groups_bn -= 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.encoder_dim, bottleneck_ch, kernel_size=1),
            nn.GroupNorm(gn_groups_bn, bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_blocks = nn.ModuleList()
        in_ch = decoder_channels[0] * 2
        
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = self.encoder_dim if i < len(skip_layers) else 0
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1], 
                              kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        )
        
        logger.info(f"Model created: {encoder_name}, classes={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        skip_features = []
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                skip_features.append(output)
            return hook
        
        hooks = []
        for idx in self.skip_layers:
            hook = self.encoder.blocks[idx].register_forward_hook(hook_fn(idx))
            hooks.append(hook)
        
        features = self.encoder.forward_features(x)
        
        for hook in hooks:
            hook.remove()
        
        # Handle dict output from newer DINOv2 versions
        if isinstance(features, dict):
            features = features['x_norm_patchtokens']
        else:
            # Handle tensor output (remove CLS token)
            features = features[:, 1:, :]
        
        h = w = int(features.shape[1] ** 0.5)
        features = features.reshape(B, h, w, -1).permute(0, 3, 1, 2)
        
        skip_features_reshaped = []
        for skip_feat in skip_features:
            # Skip features are tensors, remove CLS token
            skip_feat = skip_feat[:, 1:, :]
            skip_feat = skip_feat.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            skip_features_reshaped.append(skip_feat)
        
        x = self.bottleneck(features)
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_features_reshaped[-(i+1)] if i < len(skip_features_reshaped) else None
            x = decoder_block(x, skip)
        
        x = self.final_upsample(x)
        
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


def build_model(config: dict) -> SegmentationModel:
    model_cfg = config['model']
    return SegmentationModel(
        encoder_name=model_cfg['encoder'],
        num_classes=model_cfg['num_classes'],
        decoder_channels=model_cfg['decoder_channels'],
        skip_layers=model_cfg['skip_layers'],
        encoder_frozen=model_cfg.get('encoder_frozen', False)
    )


def build_loss(config: dict) -> CombinedLoss:
    loss_cfg = config.get('loss', {})
    class_weights = None
    if 'class_weights' in loss_cfg:
        class_weights = torch.tensor(loss_cfg['class_weights'], dtype=torch.float32)
    
    return CombinedLoss(
        dice_weight=loss_cfg.get('dice_weight', 0.5),
        focal_weight=loss_cfg.get('focal_weight', 0.5),
        focal_alpha=loss_cfg.get('focal_alpha', 0.25),
        focal_gamma=loss_cfg.get('focal_gamma', 2.0),
        class_weights=class_weights
    )


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, config):
    """Train one epoch with AMP (Mixed Precision) and Gradient Accumulation."""
    model.train()
    total_loss = 0.0
    accum_steps = config['training']['accumulation_steps']
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # AMP: Automatic Mixed Precision (Float16) -> 2x Speedup
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            # Normalize loss for accumulation
            loss = loss / accum_steps
        
        # Scalar scales loss to prevent underflow in FP16
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            # Unscale before clipping gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          config['training']['gradient_clip'])
            
            # Scaler step -> checks for Inf/NaN -> updates weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps # Log true loss magnitude
        pbar.set_postfix({'loss': f'{loss.item()*accum_steps:.4f}',
                          'eff_bs': f'{config["training"]["batch_size"] * accum_steps}'})
    
    # Handle leftover batches
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes, class_names=None):
    """Validate model and compute comprehensive metrics.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        num_classes: Number of classes
        class_names: List of class names
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    
    metrics_calculator = SegmentationMetrics(
        num_classes=num_classes,
        class_names=class_names or [f'class_{i}' for i in range(num_classes)]
    )
    
    for batch in tqdm(dataloader, desc='Validating'):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        
        pred = outputs.argmax(dim=1)
        metrics_calculator.update(pred, masks)
    
    avg_loss = total_loss / len(dataloader)
    metrics = metrics_calculator.compute()
    
    return avg_loss, metrics


print("\n" + "="*60)
print("SETUP COMPLETE - Ready to Train!")
print("="*60)
print("\nConfiguration loaded:")
print(f"  - Classes: {config['model']['num_classes']} (background, Scratch, OrangePeel)")
print(f"  - Image size: {config['data']['image_size']}x{config['data']['image_size']}")
print(f"  - Batch size: {config['training']['batch_size']}")
print(f"  - Epochs: {config['training']['num_epochs']}")
print(f"  - Learning rate: {config['training']['learning_rate']}")
print("\nNext steps:")
print("1. Update config['paths']['data_root'] to your dataset path")
print("2. Ensure dataset structure: data_root/train/images/ and data_root/train/labels/")
print("3. Run the training code below")
print("="*60)

# Update this path to your actual dataset location
# Your structure: data_root/train/images/ and data_root/train/labels/
config['paths']['data_root'] = '/kaggle/input/blackdoor'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(config).to(device)
criterion = build_loss(config)

# --- LR Scaling ---
# Linear scaling rule: when effective batch size changes, LR should scale proportionally.
# Reference: Goyal et al., "Accurate, Large Minibatch SGD" (Facebook, 2017)
# Formula: scaled_lr = base_lr * (effective_batch_size / reference_batch_size)
# This compensates for the fact that larger batches produce less noisy gradients,
# so we can take larger optimization steps.
effective_batch_size = config['training']['batch_size'] * config['training']['accumulation_steps']
reference_batch_size = config['training'].get('reference_batch_size', 32)
base_lr = config['training']['learning_rate']
scaled_lr = base_lr * (effective_batch_size / reference_batch_size)

print(f'\nLR Scaling:')
print(f'  Physical batch size:  {config["training"]["batch_size"]}')
print(f'  Accumulation steps:   {config["training"]["accumulation_steps"]}')
print(f'  Effective batch size: {effective_batch_size}')
print(f'  Reference batch size: {reference_batch_size}')
print(f'  Base LR:              {base_lr}')
print(f'  Scaled LR:            {scaled_lr}')
print(f'  Norm type:            GroupNorm (batch-independent statistics)')

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=scaled_lr,
                              weight_decay=config['training']['weight_decay'])

# Define class names for metrics
class_names = ['background', 'Chipping', 'Dust', 'Rundown', 'Scratch']

# Split dataset from single train folder
image_dir = f"{config['paths']['data_root']}/train/images"
label_dir = f"{config['paths']['data_root']}/train/labels"

print(f"Splitting dataset from: {image_dir}")
train_files, val_files, test_files = split_dataset(
    image_dir=image_dir,
    train_ratio=config['data']['train_split'],
    val_ratio=config['data']['val_split'],
    test_ratio=config['data']['test_split'],
    seed=42
)

# Training dataset
train_dataset = DefectDataset(
    image_dir=image_dir,
    label_dir=label_dir,
    image_size=config['data']['image_size'],
    transform=get_train_transform(config),
    num_classes=config['model']['num_classes'] - 1,
    is_training=True,
    image_files=train_files
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=config['data']['num_workers'],
    pin_memory=True
)

# Validation dataset
val_dataset = DefectDataset(
    image_dir=image_dir,
    label_dir=label_dir,
    image_size=config['data']['image_size'],
    transform=get_val_transform(config),
    num_classes=config['model']['num_classes'] - 1,
    is_training=False,
    image_files=val_files
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['data']['num_workers'],
    pin_memory=True
)

# Training loop with metrics
best_loss = float('inf')
best_iou = 0.0

# Initialize GradScaler for AMP
scaler = torch.cuda.amp.GradScaler()

for epoch in range(1, config['training']['num_epochs'] + 1):
    # Training
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config)
    print(f'\\nEpoch {epoch}/{config["training"]["num_epochs"]}')
    print(f'Train Loss: {train_loss:.4f}')
    
    # Progressive Unfreezing
    if epoch == config['training'].get('unfreeze_at_epoch', 0):
        print(f"\\n{'='*40}")
        print(f"🔓 UNFREEZING ENCODER (Epoch {epoch})")
        print(f"Model will now fine-tune the DINOv2 backbone.")
        print('='*40)
        for param in model.encoder.parameters():
            param.requires_grad = True
        # LR for backbone usually needs to be lower, but we'll stick to uniform for now
        # or we could scale it down:
        # optimizer.add_param_group({'params': model.encoder.parameters(), 'lr': config['training']['learning_rate'] * 0.1})
        # Since we initialized optimizer with all params (even frozen ones), we just toggle grads.
    
    # Validation with metrics (every 5 epochs or last epoch)
    if epoch % 5 == 0 or epoch == config['training']['num_epochs']:
        val_loss, metrics = validate(model, val_loader, criterion, device, 
                                     config['model']['num_classes'], class_names)
        
        print(f'Val Loss: {val_loss:.4f}')
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Mean Recall: {metrics['mean_recall']:.4f}")
        print(f"Mean Detection Accuracy: {metrics['mean_detection_accuracy']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        
        # Print per-class metrics
        for class_name in class_names:
            print(f"  {class_name} - Recall: {metrics[f'recall_{class_name}']:.4f}, "
                  f"Detection Acc: {metrics[f'detection_accuracy_{class_name}']:.4f}")
        
        # Save best model based on IoU
        if metrics['mean_iou'] > best_iou:
            best_iou = metrics['mean_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': metrics,
                'config': config
            }, 'checkpoints/best_model.pth')
            print(f'✓ Saved best model (IoU: {best_iou:.4f})')

# Final evaluation with detailed metrics
print('\\n' + '='*70)
print('FINAL EVALUATION')
print('='*70)
final_loss, final_metrics = validate(model, val_loader, criterion, device, 
                                     config['model']['num_classes'], class_names)

metrics_calc = SegmentationMetrics(config['model']['num_classes'], class_names)
metrics_calc.print_metrics(final_metrics)

import json
with open('checkpoints/final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)
print('\\n✓ Metrics saved to checkpoints/final_metrics.json')

# ==============================================================================
# EXPLAINABILITY ENGINE: Build Feature Bank & Baselines
# ==============================================================================

print('\n' + '='*70)
print('BUILDING EXPLAINABILITY FEATURE BANK')
print('='*70)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PCA_DIMS = 64   # Reduce 768-D features to 64-D for tractable Mahalanobis
NN_K = 3        # Number of nearest neighbors to retrieve

# --- Step 1: Extract DINOv2 Layer-11 features from training images ---
model.eval()
print('Extracting DINOv2 features from training images...')

all_raw_features = []   # Will collect (num_positions, 768) per image
all_feature_labels = [] # Class label per spatial position
representative_patches = {i: [] for i in range(len(class_names))}
MAX_PATCHES_PER_CLASS = 100

bank_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=config['data']['num_workers'], pin_memory=True
)

for batch in tqdm(bank_loader, desc='Feature Extraction'):
    images_bank = batch['image'].to(device)
    masks_bank = batch['mask'].numpy().squeeze(0)  # (H, W)
    img_path_bank = batch['image_path'][0]

    # Hook into encoder block 11
    _skip_feats_bank = []
    def _hook_bank(module, inp, out):
        _skip_feats_bank.append(out)
    hook_handle = model.encoder.blocks[11].register_forward_hook(_hook_bank)
    with torch.no_grad():
        _ = model.encoder.forward_features(images_bank)
    hook_handle.remove()

    feat = _skip_feats_bank[0][:, 1:, :]  # Remove CLS: (1, h*w, 768)
    feat_np = feat.squeeze(0).cpu().numpy()  # (h*w, 768)
    h_f = w_f = int(feat_np.shape[0] ** 0.5)

    mask_ds = cv2.resize(masks_bank.astype(np.uint8), (w_f, h_f),
                         interpolation=cv2.INTER_NEAREST).flatten()

    all_raw_features.append(feat_np)
    all_feature_labels.append(mask_ds)

    # Store representative image patches for NN visualization
    img_vis_bank = denormalize_for_vis(images_bank.squeeze(0))
    ppx = config['data']['image_size'] // h_f  # pixels per feature position (~14)
    for cls_i in range(len(class_names)):
        if len(representative_patches[cls_i]) >= MAX_PATCHES_PER_CLASS:
            continue
        cls_pos = np.argwhere(mask_ds == cls_i).flatten()
        if len(cls_pos) == 0:
            continue
        sample_pos = cls_pos[np.random.choice(len(cls_pos), min(5, len(cls_pos)), replace=False)]
        for sp in sample_pos:
            if len(representative_patches[cls_i]) >= MAX_PATCHES_PER_CLASS:
                break
            row_p, col_p = sp // w_f, sp % w_f
            r1p = row_p * ppx
            c1p = col_p * ppx
            r2p = min(r1p + ppx * 2, img_vis_bank.shape[0])
            c2p = min(c1p + ppx * 2, img_vis_bank.shape[1])
            if r2p - r1p > 4 and c2p - c1p > 4:
                representative_patches[cls_i].append(img_vis_bank[r1p:r2p, c1p:c2p].copy())

all_raw_features = np.concatenate(all_raw_features, axis=0)   # (N, 768)
all_feature_labels = np.concatenate(all_feature_labels, axis=0) # (N,)
print(f'Total feature vectors: {all_raw_features.shape[0]} ({all_raw_features.shape[1]}-D)')
for ci, cn in enumerate(class_names):
    print(f'  {cn}: {int((all_feature_labels == ci).sum())} vectors, '
          f'{len(representative_patches[ci])} representative patches')

# --- Step 2: PCA dimensionality reduction ---
print(f'Fitting PCA ({all_raw_features.shape[1]}-D -> {PCA_DIMS}-D)...')
pca_model = PCA(n_components=PCA_DIMS, random_state=42)
all_pca_features = pca_model.fit_transform(all_raw_features)
print(f'PCA variance retained: {pca_model.explained_variance_ratio_.sum()*100:.1f}%')

# --- Step 3: Per-class baseline statistics (mean, inv covariance) ---
print('Computing per-class Mahalanobis baselines...')
class_baseline_stats = {}
for cls_idx_s, cls_name_s in enumerate(class_names):
    cls_mask_s = (all_feature_labels == cls_idx_s)
    cls_feats_s = all_pca_features[cls_mask_s]
    if len(cls_feats_s) > PCA_DIMS + 10:
        cls_mean_s = cls_feats_s.mean(axis=0)
        cls_cov_s = np.cov(cls_feats_s, rowvar=False) + np.eye(PCA_DIMS) * 1e-5
        cls_cov_inv_s = np.linalg.inv(cls_cov_s)
        class_baseline_stats[cls_name_s] = {
            'mean': cls_mean_s, 'cov_inv': cls_cov_inv_s, 'count': int(cls_mask_s.sum())
        }
    else:
        class_baseline_stats[cls_name_s] = None
    status = '✓' if class_baseline_stats[cls_name_s] else '✗ (insufficient data)'
    print(f'  {cls_name_s}: {status}')

# --- Step 4: Classical CV baseline from background patches ---
print('Computing classical CV baseline from background regions...')
bg_edge_densities = []
bg_laplacian_vars = []
bg_texture_entropies = []
bg_gradient_mags = []
cv_sample_count = 0
MAX_CV_SAMPLES = 500

for batch_cv in bank_loader:
    if cv_sample_count >= MAX_CV_SAMPLES:
        break
    img_cv = denormalize_for_vis(batch_cv['image'].squeeze(0))
    gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    mask_cv = batch_cv['mask'].numpy().squeeze(0)
    bg_pos = np.argwhere(mask_cv == 0)
    if len(bg_pos) < 100:
        continue
    patch_sz = 32
    for _ in range(min(20, len(bg_pos) // 100)):
        ridx = np.random.randint(len(bg_pos))
        rr, cc = bg_pos[ridx]
        r1c, c1c = max(0, rr - patch_sz//2), max(0, cc - patch_sz//2)
        r2c, c2c = min(gray_cv.shape[0], r1c + patch_sz), min(gray_cv.shape[1], c1c + patch_sz)
        if r2c - r1c < 16 or c2c - c1c < 16:
            continue
        patch_cv = gray_cv[r1c:r2c, c1c:c2c]
        edges_cv = cv2.Canny(patch_cv, 50, 150)
        bg_edge_densities.append(edges_cv.sum() / (255.0 * patch_cv.size))
        bg_laplacian_vars.append(float(cv2.Laplacian(patch_cv, cv2.CV_64F).var()))
        hist_cv = cv2.calcHist([patch_cv], [0], None, [32], [0, 256]).flatten()
        hist_cv = hist_cv / (hist_cv.sum() + 1e-7)
        bg_texture_entropies.append(float(-np.sum(hist_cv[hist_cv > 0] * np.log2(hist_cv[hist_cv > 0]))))
        sx = cv2.Sobel(patch_cv, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(patch_cv, cv2.CV_64F, 0, 1, ksize=3)
        bg_gradient_mags.append(float(np.mean(np.sqrt(sx**2 + sy**2))))
        cv_sample_count += 1

bg_cv_baseline = {
    'edge_density': {'mean': float(np.mean(bg_edge_densities)), 'std': float(np.std(bg_edge_densities) + 1e-7)},
    'laplacian_var': {'mean': float(np.mean(bg_laplacian_vars)), 'std': float(np.std(bg_laplacian_vars) + 1e-7)},
    'texture_entropy': {'mean': float(np.mean(bg_texture_entropies)), 'std': float(np.std(bg_texture_entropies) + 1e-7)},
    'gradient_magnitude': {'mean': float(np.mean(bg_gradient_mags)), 'std': float(np.std(bg_gradient_mags) + 1e-7)},
}
print(f'CV baseline from {cv_sample_count} background patches:')
for m_name, m_vals in bg_cv_baseline.items():
    print(f'  {m_name}: mu={m_vals["mean"]:.4f}, sigma={m_vals["std"]:.4f}')

print('\n✓ Explainability engine ready')

# ==============================================================================
# ENHANCED TEST EVALUATION: 4-Panel Visualization + Per-Image JSON Reports
# ==============================================================================

print('\n' + '='*70)
print('GENERATING TEST VISUALIZATIONS & PER-IMAGE JSON REPORTS')
print('='*70)

# --- Configuration ---
PIXELS_PER_MM = 2.0  # Calibrate to your camera/FOV setup
# Color palette for mask visualization (matches class order)
VIS_COLORS = np.array([
    [0, 0, 0],        # 0: background - Black
    [255, 0, 0],      # 1: Chipping - Red
    [0, 255, 0],      # 2: Dust - Green
    [0, 0, 255],      # 3: Rundown - Blue
    [255, 255, 0],    # 4: Scratch - Yellow
], dtype=np.uint8)

# --- Output directories ---
vis_dir = Path('test_results/visualizations')
json_dir = Path('test_results/json_reports')
nn_dir = Path('test_results/nn_comparisons')
vis_dir.mkdir(parents=True, exist_ok=True)
json_dir.mkdir(parents=True, exist_ok=True)
nn_dir.mkdir(parents=True, exist_ok=True)

# --- Test dataset ---
test_dataset = DefectDataset(
    image_dir=image_dir,
    label_dir=label_dir,
    image_size=config['data']['image_size'],
    transform=get_val_transform(config),
    num_classes=config['model']['num_classes'] - 1,
    is_training=False,
    image_files=test_files
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,  # Process one image at a time for per-image outputs
    shuffle=False,
    num_workers=config['data']['num_workers'],
    pin_memory=True
)


def denormalize_for_vis(img_tensor):
    """Denormalize a CHW image tensor back to uint8 HWC for display."""
    img = img_tensor.cpu().numpy()
    if img.shape[0] == 3:  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def mask_to_rgb_vis(mask_np, colors=VIS_COLORS):
    """Convert a class-index mask (H, W) to an RGB image using the color palette."""
    H, W = mask_np.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx in range(len(colors)):
        rgb[mask_np == cls_idx] = colors[cls_idx]
    return rgb


def generate_4panel_image(image_np, gt_mask_np, pred_mask_np, save_path, alpha=0.5):
    """Generate and save a 4-panel visualization: Input | Ground Truth | Prediction | Error."""
    gt_rgb = mask_to_rgb_vis(gt_mask_np)
    pred_rgb = mask_to_rgb_vis(pred_mask_np)

    gt_overlay = (image_np.astype(np.float32) * (1 - alpha) + gt_rgb.astype(np.float32) * alpha).astype(np.uint8)
    pred_overlay = (image_np.astype(np.float32) * (1 - alpha) + pred_rgb.astype(np.float32) * alpha).astype(np.uint8)

    # Error map: highlight pixels where prediction != ground truth
    error_map = (gt_mask_np != pred_mask_np).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image_np)
    axes[0].set_title('Input Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gt_overlay)
    axes[1].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pred_overlay)
    axes[2].set_title('Predictions', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(error_map, cmap='Reds')
    axes[3].set_title('Error', fontsize=13, fontweight='bold')
    axes[3].axis('off')

    # Add legend for defect classes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=np.array(VIS_COLORS[i]) / 255.0,
                             label=class_names[i]) for i in range(len(class_names))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(class_names),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def compute_mahalanobis_sigma(instance_features_pca, baseline_stats):
    """Compute Mahalanobis distance (sigma score) from background baseline.

    This measures how many standard deviations a defect region's DINOv2 features
    deviate from the learned distribution of defect-free surface patches.
    Uses the full covariance structure (not just per-dimension variance).

    Math: d = sqrt( (x - mu)^T * Sigma^{-1} * (x - mu) )
    """
    if baseline_stats is None or len(instance_features_pca) == 0:
        return None
    mean_feat = instance_features_pca.mean(axis=0)
    diff = mean_feat - baseline_stats['mean']
    mahal_dist = float(np.sqrt(np.dot(np.dot(diff, baseline_stats['cov_inv']), diff)))
    return round(mahal_dist, 4)


def compute_instance_cv_metrics(image_gray, bbox, baseline):
    """Compute classical CV deviation metrics for a defect instance.

    Reports each metric's raw value, its sigma deviation from the background
    baseline, and the ratio to baseline (e.g., '3.2x higher edge density').

    Metrics:
        - Edge density: Canny edge pixel ratio (texture discontinuity)
        - Surface roughness: Laplacian variance (high-frequency content)
        - Texture entropy: Gray-level histogram entropy (disorder)
        - Gradient magnitude: Mean Sobel gradient (edge strength)
    """
    x1, y1, x2, y2 = bbox
    patch = image_gray[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return {}

    # Edge density via Canny
    edges = cv2.Canny(patch, 50, 150)
    edge_density = float(edges.sum() / (255.0 * patch.size))
    edge_sigma = round((edge_density - baseline['edge_density']['mean']) / baseline['edge_density']['std'], 2)
    edge_ratio = round(edge_density / (baseline['edge_density']['mean'] + 1e-7), 2)

    # Laplacian variance (surface roughness)
    lap_var = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    lap_sigma = round((lap_var - baseline['laplacian_var']['mean']) / baseline['laplacian_var']['std'], 2)
    lap_ratio = round(lap_var / (baseline['laplacian_var']['mean'] + 1e-7), 2)

    # Texture entropy
    hist = cv2.calcHist([patch], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    tex_entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
    tex_sigma = round((tex_entropy - baseline['texture_entropy']['mean']) / baseline['texture_entropy']['std'], 2)

    # Gradient magnitude
    sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))
    grad_sigma = round((grad_mag - baseline['gradient_magnitude']['mean']) / baseline['gradient_magnitude']['std'], 2)
    grad_ratio = round(grad_mag / (baseline['gradient_magnitude']['mean'] + 1e-7), 2)

    return {
        'edge_density': {'value': round(edge_density, 4), 'sigma_deviation': edge_sigma,
                         'ratio_to_baseline': edge_ratio},
        'surface_roughness': {'value': round(lap_var, 2), 'sigma_deviation': lap_sigma,
                              'ratio_to_baseline': lap_ratio},
        'texture_entropy': {'value': round(tex_entropy, 4), 'sigma_deviation': tex_sigma},
        'gradient_magnitude': {'value': round(grad_mag, 2), 'sigma_deviation': grad_sigma,
                               'ratio_to_baseline': grad_ratio}
    }


def find_knn(query_feat_pca, bank_feats, bank_labels, cls_names, k=3):
    """Find K nearest neighbors by cosine similarity in PCA feature space.

    Returns the class labels and distances of the K most similar training patches.
    This gives engineers visual evidence of what the model is comparing against.
    """
    query_norm = query_feat_pca / (np.linalg.norm(query_feat_pca) + 1e-7)
    bank_norms = bank_feats / (np.linalg.norm(bank_feats, axis=1, keepdims=True) + 1e-7)
    similarities = np.dot(bank_norms, query_norm)
    top_k = np.argsort(similarities)[-k:][::-1]
    return [{
        'class': cls_names[int(bank_labels[idx])],
        'similarity': round(float(similarities[idx]), 4),
        'distance': round(float(1.0 - similarities[idx]), 4)
    } for idx in top_k]


def generate_nn_comparison(query_patch, nn_patches_list, nn_info, save_path):
    """Generate a visual comparison strip: flagged defect patch vs. nearest neighbors."""
    n_panels = 1 + len(nn_patches_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    axes[0].imshow(query_patch)
    axes[0].set_title('Query (Flagged)', fontsize=11, fontweight='bold', color='red')
    axes[0].axis('off')
    for i, (nn_patch, info) in enumerate(zip(nn_patches_list, nn_info)):
        axes[i+1].imshow(nn_patch)
        color = 'green' if info['class'] != 'background' else 'gray'
        axes[i+1].set_title(f"NN{i+1}: {info['class']}\nd={info['distance']:.3f}",
                            fontsize=10, color=color)
        axes[i+1].axis('off')
    plt.suptitle('Nearest Neighbor Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def build_explanation_text(sigma_score, cv_metrics, nn_info, defect_class):
    """Build a human-readable explanation string for engineers.

    Example output:
        'Classified as "scratch". Feature deviation: 3.2sigma from defect-free
         baseline. Primary indicator: surface_roughness (4.1sigma from normal).
         Most similar to: scratch, scratch, background in training data.'
    """
    parts = [f'Classified as "{defect_class}"']
    if sigma_score is not None:
        parts.append(f'Feature deviation: {sigma_score}\u03c3 from defect-free baseline')
    if cv_metrics:
        max_metric, max_sigma = None, 0
        for metric, vals in cv_metrics.items():
            if abs(vals.get('sigma_deviation', 0)) > abs(max_sigma):
                max_sigma = vals['sigma_deviation']
                max_metric = metric
        if max_metric:
            parts.append(f'Primary indicator: {max_metric} ({max_sigma}\u03c3 from normal)')
    if nn_info:
        nn_classes = [n['class'] for n in nn_info]
        nn_classes_str = ', '.join(nn_classes)
        parts.append(f'Most similar to: {nn_classes_str} in training data')
    return '. '.join(parts) + '.'


def generate_per_image_json(image_path_str, pred_mask_np, prob_map_np, class_names_list,
                            pixels_per_mm=PIXELS_PER_MM):
    """
    Generate a per-image JSON report with defect instance analysis.

    Args:
        image_path_str: Absolute path to the source image
        pred_mask_np: (H, W) predicted class indices
        prob_map_np: (C, H, W) softmax probability map
        class_names_list: list of class names (index 0 = background)
        pixels_per_mm: calibration factor

    Returns:
        dict matching the user's specified JSON format
    """
    H, W = pred_mask_np.shape
    total_pixels = H * W
    px_to_mm2 = 1.0 / (pixels_per_mm ** 2)

    # Defect classes are indices 1..N (skip background at 0)
    defect_class_names = class_names_list[1:]  # e.g. ['Chipping', 'Dust', 'Rundown', 'Scratch']

    total_defect_count = 0
    total_defect_area_px = 0
    defects_dict = {}

    for cls_idx, cls_name in enumerate(defect_class_names, start=1):
        # Binary mask for this class
        cls_mask = (pred_mask_np == cls_idx).astype(np.uint8)

        if cls_mask.sum() == 0:
            defects_dict[cls_name.lower()] = {
                "count": 0,
                "total_area_pixels": 0,
                "total_area_mm2": 0.0,
                "area_percentage_of_image": 0.0,
                "area_percentage_of_all_defects": 0.0,
                "instances": []
            }
            continue

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cls_mask, connectivity=8
        )

        instances = []
        cls_total_area_px = 0

        for inst_id in range(1, num_labels):  # Skip label 0 (background of this binary mask)
            x = int(stats[inst_id, cv2.CC_STAT_LEFT])
            y = int(stats[inst_id, cv2.CC_STAT_TOP])
            w = int(stats[inst_id, cv2.CC_STAT_WIDTH])
            h = int(stats[inst_id, cv2.CC_STAT_HEIGHT])
            area_px = int(stats[inst_id, cv2.CC_STAT_AREA])
            cx = round(float(centroids[inst_id][0]), 2)
            cy = round(float(centroids[inst_id][1]), 2)

            # Uncertainty: mean(1 - max_prob) over instance pixels
            inst_mask = (labels == inst_id)
            max_probs = prob_map_np.max(axis=0)  # (H, W) max prob per pixel
            uncertainty = round(float(np.mean(1.0 - max_probs[inst_mask])), 4)

            area_mm2 = round(area_px * px_to_mm2, 4)
            cls_total_area_px += area_px

            instances.append({
                "instance_id": len(instances) + 1,
                "bbox": [x, y, x + w, y + h],
                "area_pixels": area_px,
                "area_mm2": area_mm2,
                "centroid": [cx, cy],
                "uncertainty": uncertainty
            })

        total_defect_count += len(instances)
        total_defect_area_px += cls_total_area_px

        defects_dict[cls_name.lower()] = {
            "count": len(instances),
            "total_area_pixels": cls_total_area_px,
            "total_area_mm2": round(cls_total_area_px * px_to_mm2, 4),
            "area_percentage_of_image": round(cls_total_area_px / total_pixels * 100, 4),
            "area_percentage_of_all_defects": 0.0,  # computed after all classes
            "instances": instances
        }

    # Compute area_percentage_of_all_defects
    if total_defect_area_px > 0:
        for cls_name_lower in defects_dict:
            defects_dict[cls_name_lower]["area_percentage_of_all_defects"] = round(
                defects_dict[cls_name_lower]["total_area_pixels"] / total_defect_area_px * 100, 4
            )

    report = {
        "image": image_path_str,
        "summary": {
            "total_defects": total_defect_count,
            "total_defect_area_pixels": total_defect_area_px,
            "total_defect_area_mm2": round(total_defect_area_px * px_to_mm2, 4),
            "defect_area_percentage": round(total_defect_area_px / total_pixels * 100, 4)
        },
        "defects": defects_dict
    }

    return report


# --- Run Test Evaluation with Explainability ---
model.eval()
test_pca_features_all = []
test_pca_labels_all = []
print(f'\nProcessing {len(test_dataset)} test images with explainability...')

for batch_idx, batch in enumerate(tqdm(test_loader, desc='Test Evaluation')):
    images = batch['image'].to(device)
    masks = batch['mask']
    img_path = batch['image_path'][0]  # batch_size=1

    # Hook into encoder block 11 to capture features for explainability
    _test_skip_feats = []
    def _test_hook(module, inp, out):
        _test_skip_feats.append(out)
    hook_h = model.encoder.blocks[11].register_forward_hook(_test_hook)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)  # (1, C, H, W)

    hook_h.remove()

    pred_mask = outputs.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
    gt_mask = masks.squeeze(0).cpu().numpy()  # (H, W)
    prob_map = probs.squeeze(0).cpu().numpy()  # (C, H, W)

    # Extract & PCA-transform test features
    test_feat = _test_skip_feats[0][:, 1:, :].squeeze(0).cpu().numpy()  # (h*w, 768)
    h_feat = w_feat = int(test_feat.shape[0] ** 0.5)
    test_feat_pca = pca_model.transform(test_feat)  # (h*w, PCA_DIMS)
    test_mask_ds = cv2.resize(pred_mask.astype(np.uint8), (w_feat, h_feat),
                              interpolation=cv2.INTER_NEAREST).flatten()
    test_pca_features_all.append(test_feat_pca)
    test_pca_labels_all.append(test_mask_ds)

    # Denormalize image for visualization
    image_np = denormalize_for_vis(images.squeeze(0))
    gray_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    img_stem = Path(img_path).stem

    # 1. Generate 4-panel visualization image
    vis_save_path = vis_dir / f'{img_stem}_result.png'
    generate_4panel_image(image_np, gt_mask, pred_mask, str(vis_save_path))

    # 2. Generate per-image JSON report
    report = generate_per_image_json(img_path, pred_mask, prob_map, class_names)

    # 3. Enhance each defect instance with explainability
    ppx_test = config['data']['image_size'] // h_feat
    for cls_name_lower, cls_data in report['defects'].items():
        cls_idx_e = next((i for i, n in enumerate(class_names) if n.lower() == cls_name_lower), None)
        if cls_idx_e is None or cls_data['count'] == 0:
            continue

        for inst in cls_data['instances']:
            bbox = inst['bbox']  # [x1, y1, x2, y2]

            # Gather feature vectors overlapping this instance
            fr1 = max(0, bbox[1] // ppx_test)
            fc1 = max(0, bbox[0] // ppx_test)
            fr2 = min(h_feat, bbox[3] // ppx_test + 1)
            fc2 = min(w_feat, bbox[2] // ppx_test + 1)
            inst_feat_idx = [r * w_feat + c for r in range(fr1, fr2) for c in range(fc1, fc2)]

            if len(inst_feat_idx) == 0:
                inst['explainability'] = {'error': 'No feature vectors overlap this instance'}
                continue

            inst_feats = test_feat_pca[inst_feat_idx]

            # a) Mahalanobis sigma-score from background baseline
            sigma_score = compute_mahalanobis_sigma(inst_feats, class_baseline_stats.get('background'))

            # b) Classical CV metrics with deviation from baseline
            cv_metrics = compute_instance_cv_metrics(gray_np, bbox, bg_cv_baseline)

            # c) K nearest neighbors in feature space
            mean_inst_feat = inst_feats.mean(axis=0)
            nn_info = find_knn(mean_inst_feat, all_pca_features, all_feature_labels, class_names, k=NN_K)

            # d) Build human-readable explanation
            explanation = build_explanation_text(sigma_score, cv_metrics, nn_info, cls_name_lower)

            # e) Top deviating features (sorted by |sigma|)
            top_devs = []
            if cv_metrics:
                sorted_m = sorted(cv_metrics.items(),
                                  key=lambda x: abs(x[1].get('sigma_deviation', 0)), reverse=True)
                top_devs = [{'feature': n, 'deviation_sigma': v['sigma_deviation']}
                            for n, v in sorted_m[:3]]

            inst['explainability'] = {
                'sigma_score': sigma_score,
                'mahalanobis_distance': sigma_score,
                'explanation': explanation,
                'top_deviating_features': top_devs,
                'nearest_neighbors': nn_info,
                'cv_metrics': cv_metrics
            }

            # f) Generate NN comparison image (first 3 instances per image)
            if inst['instance_id'] <= 3:
                qpatch = image_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if qpatch.size > 0 and qpatch.shape[0] > 2 and qpatch.shape[1] > 2:
                    nn_patches_vis = []
                    for nn in nn_info:
                        nn_cls_i = next((i for i, n in enumerate(class_names) if n == nn['class']), 0)
                        if representative_patches[nn_cls_i]:
                            nn_patches_vis.append(representative_patches[nn_cls_i][0])
                        else:
                            nn_patches_vis.append(np.zeros((32, 32, 3), dtype=np.uint8))
                    nn_save_path = nn_dir / f'{img_stem}_inst{inst["instance_id"]}_nn.png'
                    generate_nn_comparison(qpatch, nn_patches_vis, nn_info, str(nn_save_path))

    # Save enhanced JSON report
    json_save_path = json_dir / f'{img_stem}_report.json'
    with open(json_save_path, 'w') as f:
        json.dump(report, f, indent=2)

# ==============================================================================
# Feature Space Visualization (t-SNE)
# ==============================================================================
print('\nGenerating feature space visualization...')
test_pca_all = np.concatenate(test_pca_features_all, axis=0)
test_labels_all = np.concatenate(test_pca_labels_all, axis=0)

# Subsample for t-SNE
tsne_max_train = min(3000, len(all_pca_features))
tsne_max_test = min(1000, len(test_pca_all))
train_sub_idx = np.random.choice(len(all_pca_features), tsne_max_train, replace=False)
test_sub_idx = np.random.choice(len(test_pca_all), tsne_max_test, replace=False)

combined_feats = np.concatenate([all_pca_features[train_sub_idx], test_pca_all[test_sub_idx]])
combined_labels = np.concatenate([all_feature_labels[train_sub_idx], test_labels_all[test_sub_idx]])
combined_split = np.array([0]*len(train_sub_idx) + [1]*len(test_sub_idx))  # 0=train, 1=test

print(f'Running t-SNE on {len(combined_feats)} vectors...')
perplexity = min(30, len(combined_feats) - 1)
tsne_model = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
features_2d = tsne_model.fit_transform(combined_feats)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Panel 1: Feature space colored by class
for cls_idx_v, cls_name_v in enumerate(class_names):
    m = combined_labels == cls_idx_v
    if m.sum() > 0:
        c = np.array(VIS_COLORS[cls_idx_v]) / 255.0 if cls_idx_v > 0 else [0.7, 0.7, 0.7]
        axes[0].scatter(features_2d[m, 0], features_2d[m, 1], c=[c], s=5, alpha=0.4, label=cls_name_v)
axes[0].set_title('Feature Space by Defect Class', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9, markerscale=3)
axes[0].set_xlabel('t-SNE Dim 1')
axes[0].set_ylabel('t-SNE Dim 2')
axes[0].grid(True, alpha=0.2)

# Panel 2: Train vs Test distribution
axes[1].scatter(features_2d[combined_split == 0, 0], features_2d[combined_split == 0, 1],
                c='steelblue', s=5, alpha=0.3, label='Train')
axes[1].scatter(features_2d[combined_split == 1, 0], features_2d[combined_split == 1, 1],
                c='crimson', s=10, alpha=0.6, label='Test', marker='x')
axes[1].set_title('Train vs Test Distribution', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=9, markerscale=3)
axes[1].set_xlabel('t-SNE Dim 1')
axes[1].set_ylabel('t-SNE Dim 2')
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('test_results/feature_space_visualization.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'\n' + '='*70)
print('ALL OUTPUTS GENERATED SUCCESSFULLY')
print('='*70)
print(f'\n✓ 4-panel visualizations       -> {vis_dir}')
print(f'✓ JSON reports (+ explainability) -> {json_dir}')
print(f'✓ NN comparison images           -> {nn_dir}')
print(f'✓ Feature space t-SNE plot       -> test_results/feature_space_visualization.png')
print(f'✓ Final metrics                  -> checkpoints/final_metrics.json')
print(f'\nTotal test images processed: {len(test_dataset)}')
print('\nExplainability features per defect instance:')
print('  - Mahalanobis sigma-score (deviation from defect-free baseline)')
print('  - CV metrics: edge density, surface roughness, texture entropy, gradient magnitude')
print('  - K nearest neighbor comparison with training patches')
print('  - Human-readable explanation text')
print('='*70)