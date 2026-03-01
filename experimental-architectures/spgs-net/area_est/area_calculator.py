import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CalibrationConfig


class AreaCalculator:
    def __init__(
        self,
        mm_per_pixel: float = None,
        calibration_file: Optional[str] = None
    ):
        if calibration_file and Path(calibration_file).exists():
            self.mm_per_pixel = self._load_calibration(calibration_file)
        else:
            self.mm_per_pixel = mm_per_pixel or CalibrationConfig.MM_PER_PIXEL
        
        self.mm2_per_pixel2 = self.mm_per_pixel ** 2
        
        print(f"[Section 7] Area calculator initialized:")
        print(f"  mm_per_pixel: {self.mm_per_pixel}")
        print(f"  mm²_per_pixel²: {self.mm2_per_pixel2:.6f}")
    
    def _load_calibration(self, calibration_file: str) -> float:
        import json
        
        with open(calibration_file, 'r') as f:
            calib = json.load(f)
        
        if 'mm_per_pixel' in calib:
            return float(calib['mm_per_pixel'])
        if 'reference_length_mm' in calib and 'reference_length_pixels' in calib:
            return calib['reference_length_mm'] / calib['reference_length_pixels']
        
        raise ValueError(f"Invalid calibration file format: {calibration_file}")
    
    def pixel_to_mm2(self, pixel_area: Union[int, float]) -> float:
        return float(pixel_area) * self.mm2_per_pixel2
    
    def calculate_instance_area(
        self,
        instance: Dict
    ) -> Dict:
        pixel_area = instance.get('area_pixels', 0)
        instance['area_mm2'] = self.pixel_to_mm2(pixel_area)
        return instance
    
    def process_instances(
        self,
        instances: List[Dict]
    ) -> List[Dict]:
        return [self.calculate_instance_area(inst) for inst in instances]
    
    def calculate_from_mask(
        self,
        mask: np.ndarray
    ) -> float:
        pixel_count = mask.sum()
        return self.pixel_to_mm2(pixel_count)
    
    def get_calibration_info(self) -> Dict:
        return {
            'mm_per_pixel': self.mm_per_pixel,
            'mm2_per_pixel2': self.mm2_per_pixel2,
            'note': 'Values from camera calibration (see Section 7)'
        }


def calculate_defect_area(
    mask_or_pixel_count: Union[np.ndarray, int, float],
    mm_per_pixel: float = None
) -> float:
    mm_per_pixel = mm_per_pixel or CalibrationConfig.MM_PER_PIXEL
    mm2_per_pixel2 = mm_per_pixel ** 2
    
    if isinstance(mask_or_pixel_count, np.ndarray):
        pixel_count = mask_or_pixel_count.sum()
    else:
        pixel_count = mask_or_pixel_count
    
    return float(pixel_count) * mm2_per_pixel2


class CalibrationHelper:
    
    @staticmethod
    def calibrate_from_reference(
        reference_length_mm: float,
        reference_length_pixels: int
    ) -> float:
        return reference_length_mm / reference_length_pixels
    
    @staticmethod
    def calibrate_from_grid(
        grid_spacing_mm: float,
        grid_points: List[Tuple[int, int]],
        axis: str = 'x'
    ) -> float:
        if len(grid_points) < 2:
            raise ValueError("Need at least 2 grid points")
        idx = 0 if axis == 'x' else 1
        sorted_points = sorted(grid_points, key=lambda p: p[idx])
        distances = []
        for i in range(len(sorted_points) - 1):
            dist = abs(sorted_points[i + 1][idx] - sorted_points[i][idx])
            distances.append(dist)
        avg_pixel_dist = np.mean(distances)
        
        return grid_spacing_mm / avg_pixel_dist
    
    @staticmethod
    def save_calibration(
        mm_per_pixel: float,
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        import json
        from datetime import datetime
        
        calib = {
            'mm_per_pixel': mm_per_pixel,
            'mm2_per_pixel2': mm_per_pixel ** 2,
            'calibration_date': datetime.now().isoformat(),
        }
        
        if metadata:
            calib.update(metadata)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(calib, f, indent=2)
        
        print(f"[Section 7] Calibration saved to {output_path}")


if __name__ == "__main__":
    print("Testing Area Calculator...")
    print("=" * 60)
    
    calculator = AreaCalculator()
    
    test_pixel_areas = [100, 500, 1000, 5000]
    
    print("\n[Section 7] Pixel to mm² conversion:")
    for pixels in test_pixel_areas:
        mm2 = calculator.pixel_to_mm2(pixels)
        print(f"  {pixels} pixels = {mm2:.4f} mm²")
    
    dummy_instance = {
        'class_id': 1,
        'class_name': 'Dust',
        'area_pixels': 2500,
        'bbox': [100, 100, 150, 150]
    }
    
    result = calculator.calculate_instance_area(dummy_instance)
    print(f"\n[Section 7] Instance area calculation:")
    print(f"  Input: {dummy_instance['area_pixels']} pixels")
    print(f"  Output: {result['area_mm2']:.4f} mm²")
    
    # Test calibration helper
    print("\n[Section 7] Calibration example:")
    # Example: A 10mm reference measures 100 pixels
    calc_mm_per_pixel = CalibrationHelper.calibrate_from_reference(
        reference_length_mm=10.0,
        reference_length_pixels=100
    )
    print(f"  Reference: 10mm = 100px")
    print(f"  Calculated mm_per_pixel: {calc_mm_per_pixel}")
    
    # Test from mask
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[25:75, 25:75] = 1  # 50x50 = 2500 pixels
    
    area_from_mask = calculator.calculate_from_mask(dummy_mask)
    print(f"\n[Section 7] Area from 50x50 mask: {area_from_mask:.4f} mm²")
