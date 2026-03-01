import cv2
import numpy as np

img  = cv2.imread("keyhole/grid_4_8.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

otsu_thresh, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dark_thresh = otsu_thresh * 0.7
_, dark_mask = cv2.threshold(blurred, dark_thresh, 255, cv2.THRESH_BINARY_INV)

grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel  = np.sqrt(grad_x**2 + grad_y**2)
sobel  = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, sobel_bin = cv2.threshold(sobel, 80, 255, cv2.THRESH_BINARY)  

canny = cv2.Canny(blurred, threshold1=20, threshold2=150)  

edges_fused = cv2.bitwise_or(sobel_bin, canny)
kernel_edge = np.ones((3, 3), np.uint8)
edges_fused = cv2.dilate(edges_fused, kernel_edge, iterations=1)

hole_interior = cv2.bitwise_and(dark_mask, cv2.bitwise_not(edges_fused))

kernel_open  = np.ones((7, 7), np.uint8)   
kernel_close = np.ones((13, 13), np.uint8)  
hole_interior = cv2.morphologyEx(hole_interior, cv2.MORPH_OPEN,  kernel_open)
hole_interior = cv2.morphologyEx(hole_interior, cv2.MORPH_CLOSE, kernel_close)

contours, _ = cv2.findContours(hole_interior, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

hole_mask = np.zeros_like(gray)

img_area    = gray.shape[0] * gray.shape[1]
MIN_AREA    = 300
MAX_AREA    = img_area
MIN_CIRC    = 0.4
PX_TO_MM2   = 0.0037
kept = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        continue
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if circularity < MIN_CIRC:
        continue
    kept.append(cnt)
    cv2.drawContours(hole_mask, [cnt], -1, 255, thickness=cv2.FILLED)

overlay = img.copy()
overlay[hole_mask == 255] = (0, 0, 255)
result  = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

cv2.drawContours(result, kept, -1, (0, 255, 0), 2)

for i, cnt in enumerate(kept):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    area_px = cv2.contourArea(cnt)
    area_mm2 = area_px * PX_TO_MM2
    cv2.putText(result, f"H{i+1} {area_mm2:.2f}mm2",
                (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1, cv2.LINE_AA)

cv2.imwrite("hybrid_dark_mask.png",   dark_mask)
cv2.imwrite("hybrid_edges_fused.png", edges_fused)
cv2.imwrite("hybrid_hole_mask.png",   hole_mask)
cv2.imwrite("hybrid_result.png",      result)

print(f"Otsu threshold: {otsu_thresh:.1f}  dark_thresh used: {dark_thresh:.1f}")
print(f"MAX_AREA limit: {int(MAX_AREA)} px²")
print(f"Detected {len(kept)} hole(s)")
for i, cnt in enumerate(kept):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    c = round(4 * 3.14159 * a / (p ** 2), 2) if p > 0 else 0
    mm2 = a * PX_TO_MM2
    print(f"  Hole {i+1}: area={int(a)} px²  ({mm2:.2f} mm²)  circularity={c}")
