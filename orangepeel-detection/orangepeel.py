import cv2
import numpy as np

img  = cv2.imread("train/images/grid_2_14_jpg.rf.10f5f81cd40d0d8666a5552599a500cf.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
log_abs = np.abs(log)
log_norm = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, rough_mask = cv2.threshold(log_norm, 15, 255, cv2.THRESH_BINARY)

kernel_open  = np.ones((3, 3), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)
rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN,  kernel_open)
rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, kernel_close)

contours, _ = cv2.findContours(rough_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

img_area = gray.shape[0] * gray.shape[1]
MIN_AREA = 900    
MAX_AREA = img_area * 0.5  
kept = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        continue
    kept.append(cnt)

PX_TO_MM2      = 0.0037   
THRESHOLD_MM2  = 5.0      

total_area_px  = sum(cv2.contourArea(c) for c in kept)
total_area_mm2 = total_area_px * PX_TO_MM2

result = img.copy()

if total_area_mm2 > THRESHOLD_MM2:
    orange_overlay = np.full_like(img, (0, 165, 255))
    result = cv2.addWeighted(img, 0.35, orange_overlay, 0.65, 0)

    h_img, w_img = img.shape[:2]
    label      = "Orange Peel"
    area_label = f"Total: {total_area_mm2:.2f} mm2"
    font       = cv2.FONT_HERSHEY_DUPLEX
    (lw, lh), _ = cv2.getTextSize(label, font, 1.8, 3)
    cv2.putText(result, label,
                ((w_img - lw) // 2, h_img // 2),
                font, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
    (aw, _), _ = cv2.getTextSize(area_label, font, 0.9, 2)
    cv2.putText(result, area_label,
                ((w_img - aw) // 2, h_img // 2 + lh + 10),
                font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

    verdict = f"ORANGE PEEL  ({total_area_mm2:.2f} mm² > {THRESHOLD_MM2} mm² threshold)"
else:
    semantic_mask = np.zeros_like(gray)
    for cnt in kept:
        cv2.drawContours(semantic_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    result[semantic_mask == 255] = (0, 165, 255)
    result = cv2.addWeighted(img, 0.45, result, 0.55, 0)

    for i, cnt in enumerate(kept):
        x, y, w, h  = cv2.boundingRect(cnt)
        area_mm2    = cv2.contourArea(cnt) * PX_TO_MM2
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"OP{i+1} {area_mm2:.2f}mm2",
                    (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 0), 1, cv2.LINE_AA)

    verdict = f"OK  ({total_area_mm2:.2f} mm² <= {THRESHOLD_MM2} mm² threshold)"

cv2.imwrite("op_clahe_enhanced.png", enhanced)
cv2.imwrite("op_log_response.png",   log_norm)
cv2.imwrite("op_rough_mask.png",     rough_mask)
cv2.imwrite("op_result.png",         result)

print(f"Detected {len(kept)} region(s)  |  Total area: {total_area_mm2:.2f} mm²")
print(f"Verdict : {verdict}")
for i, cnt in enumerate(kept):
    x, y, w, h = cv2.boundingRect(cnt)
    a   = cv2.contourArea(cnt)
    mm2 = a * PX_TO_MM2
    print(f"  Region {i+1}: bbox=({x},{y},{w},{h})  area={int(a)} px²  ({mm2:.2f} mm²)")
