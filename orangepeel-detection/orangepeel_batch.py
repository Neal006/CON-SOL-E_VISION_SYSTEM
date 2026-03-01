import cv2
import numpy as np
import glob
import os

INPUT_DIR     = "train/images"
OUTPUT_DIR    = "op_results"
PX_TO_MM2     = 0.0037
THRESHOLD_MM2 = 4.0   
MIN_AREA      = 300
CLAHE_CLIP    = 3.0
CLAHE_TILE    = (8, 8)
LOG_THRESH    = 15
OPEN_K        = 3
CLOSE_K       = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_PATHS = sorted(glob.glob(f"{INPUT_DIR}/*.jpg") +
                     glob.glob(f"{INPUT_DIR}/*.png"))
print(f"Found {len(IMAGE_PATHS)} images in {INPUT_DIR}/\n")

def process(img_path):
    img  = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enhanced = clahe.apply(gray)

    blurred  = cv2.GaussianBlur(enhanced, (3, 3), 0)
    log      = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    log_norm = cv2.normalize(np.abs(log), None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)

    _, rough = cv2.threshold(log_norm, LOG_THRESH, 255, cv2.THRESH_BINARY)
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN,
                              np.ones((OPEN_K, OPEN_K), np.uint8))
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE,
                              np.ones((CLOSE_K, CLOSE_K), np.uint8))

    img_area = gray.shape[0] * gray.shape[1]
    cnts, _  = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in cnts if MIN_AREA <= cv2.contourArea(c) <= img_area * 0.5]

    total_px   = sum(cv2.contourArea(c) for c in kept)
    total_mm2  = total_px * PX_TO_MM2
    flagged    = total_mm2 > THRESHOLD_MM2

    result = img.copy()
    if flagged:
        orange = np.full_like(img, (0, 165, 255))
        result = cv2.addWeighted(img, 0.35, orange, 0.65, 0)
        h, w   = img.shape[:2]
        font   = cv2.FONT_HERSHEY_DUPLEX
        label  = "Orange Peel"
        sub    = f"{total_mm2:.2f} mm2"
        (lw, lh), _ = cv2.getTextSize(label, font, 1.6, 3)
        cv2.putText(result, label,
                    ((w - lw) // 2, h // 2),
                    font, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        (sw, _), _  = cv2.getTextSize(sub, font, 0.8, 2)
        cv2.putText(result, sub,
                    ((w - sw) // 2, h // 2 + lh + 8),
                    font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        sem = np.zeros_like(gray)
        for c in kept:
            cv2.drawContours(sem, [c], -1, 255, cv2.FILLED)
        result[sem == 255] = (0, 165, 255)
        result = cv2.addWeighted(img, 0.45, result, 0.55, 0)
        for i, c in enumerate(kept):
            x, y, w2, h2 = cv2.boundingRect(c)
            mm2 = cv2.contourArea(c) * PX_TO_MM2
            cv2.rectangle(result, (x, y), (x+w2, y+h2), (0, 255, 0), 2)
            cv2.putText(result, f"OP{i+1} {mm2:.2f}mm2",
                        (x, max(y-4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 0), 1, cv2.LINE_AA)

    return result, len(kept), total_mm2, flagged

header = f"{'#':>3}  {'Image':<45}  {'Regions':>7}  {'Total mm²':>10}  {'Verdict'}"
print(header)
print("─" * len(header))

flagged_count = 0
for idx, path in enumerate(IMAGE_PATHS, 1):
    out = process(path)
    if out is None:
        print(f"{idx:>3}  {os.path.basename(path):<45}  ERROR")
        continue

    result, n_regions, total_mm2, flagged = out
    verdict = "ORANGE PEEL ⚠" if flagged else "OK"
    if flagged:
        flagged_count += 1

    out_name = os.path.splitext(os.path.basename(path))[0] + "_result.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), result)

    print(f"{idx:>3}  {os.path.basename(path):<45}  {n_regions:>7}  {total_mm2:>10.2f}  {verdict}")

print("─" * len(header))
print(f"\nTotal images  : {len(IMAGE_PATHS)}")
print(f"Flagged       : {flagged_count}  ({100*flagged_count/len(IMAGE_PATHS):.1f}%)")
print(f"Clean         : {len(IMAGE_PATHS)-flagged_count}")
print(f"\nAnnotated results saved to: {OUTPUT_DIR}/")
