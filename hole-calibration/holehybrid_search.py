import cv2
import numpy as np
import itertools

img  = cv2.imread("keyhole/grid_1_7.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
img_area = gray.shape[0] * gray.shape[1]

otsu_thresh, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
DARK_THRESH_RATIO = 0.7
MIN_CIRC          = 0.4

dark_thresh = otsu_thresh * DARK_THRESH_RATIO
_, dark_mask = cv2.threshold(blurred, dark_thresh, 255, cv2.THRESH_BINARY_INV)

SOBEL_THRESH_VALUES   = [20, 40, 60, 80]
CANNY_LOW_VALUES      = [20, 30, 50]
CANNY_HIGH_VALUES     = [80, 100, 150]
OPEN_KERNEL_VALUES    = [3, 5, 7]
CLOSE_KERNEL_VALUES   = [5, 9, 13]
MIN_AREA_VALUES       = [200, 500, 1000]
MAX_AREA_RATIO_VALUES = [0.01, 0.03, 0.10]

print(f"Otsu={otsu_thresh:.1f}  dark_thresh={dark_thresh:.1f}  MIN_CIRC={MIN_CIRC}")
print(f"Image area: {img_area} px²")
total_combos = (len(SOBEL_THRESH_VALUES) * len(CANNY_LOW_VALUES) *
                len(CANNY_HIGH_VALUES)   * len(OPEN_KERNEL_VALUES) *
                len(CLOSE_KERNEL_VALUES) * len(MIN_AREA_VALUES) *
                len(MAX_AREA_RATIO_VALUES))
print(f"Total combinations: {total_combos}\n")

header = (f"{'Sobel':>5} {'CnyLo':>5} {'CnyHi':>5} "
          f"{'Open':>4} {'Close':>5} {'MinA':>5} {'MaxA%':>5} "
          f"{'Holes':>5}  Details")
print(header)
print("-" * len(header))

results = []

for (sobel_t, canny_lo, canny_hi,
     open_k, close_k, min_area, max_area_ratio) in itertools.product(
        SOBEL_THRESH_VALUES, CANNY_LOW_VALUES, CANNY_HIGH_VALUES,
        OPEN_KERNEL_VALUES,  CLOSE_KERNEL_VALUES,
        MIN_AREA_VALUES,     MAX_AREA_RATIO_VALUES):

    if canny_lo >= canny_hi:
        continue

    max_area = img_area * max_area_ratio
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel  = np.sqrt(grad_x**2 + grad_y**2)
    sobel  = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sobel_bin = cv2.threshold(sobel, sobel_t, 255, cv2.THRESH_BINARY)

    canny = cv2.Canny(blurred, canny_lo, canny_hi)

    edges_fused = cv2.bitwise_or(sobel_bin, canny)
    kernel_edge = np.ones((3, 3), np.uint8)
    edges_fused = cv2.dilate(edges_fused, kernel_edge, iterations=1)

    hole_interior = cv2.bitwise_and(dark_mask, cv2.bitwise_not(edges_fused))

    ko = np.ones((open_k,  open_k),  np.uint8)
    kc = np.ones((close_k, close_k), np.uint8)
    hole_interior = cv2.morphologyEx(hole_interior, cv2.MORPH_OPEN,  ko)
    hole_interior = cv2.morphologyEx(hole_interior, cv2.MORPH_CLOSE, kc)

    contours, _ = cv2.findContours(hole_interior, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circ = 4 * np.pi * area / (perimeter ** 2)
        if circ < MIN_CIRC:
            continue
        kept.append((area, circ))

    n_holes = len(kept)
    detail = "  ".join(
        f"[area={int(a)} circ={c:.2f}]" for a, c in kept
    ) if kept else "-"

    results.append((n_holes, sobel_t, canny_lo, canny_hi,
                    open_k, close_k, min_area, max_area_ratio, detail))

def sort_key(r):
    n, st, clo, chi, ok, ck, mina, maxar, det = r
    import re
    circs = [float(x) for x in re.findall(r'circ=(\d+\.\d+)', det)]
    avg_circ = sum(circs) / len(circs) if circs else 0.0
    return (-n, -avg_circ)

results.sort(key=sort_key)

for (n, st, clo, chi, ok, ck, mina, maxar, det) in results:
    print(f"{st:>5} {clo:>5} {chi:>5} "
          f"{ok:>4} {ck:>5} {mina:>5} {maxar*100:>4.0f}% "
          f"{n:>5}  {det}")

print(f"\nDone. {len(results)} valid combinations evaluated.")

with open("best_params.txt", "w") as f:
    f.write(f"Grid Search Best Results\n")
    f.write(f"Fixed: dark_thresh=otsu*{DARK_THRESH_RATIO}  MIN_CIRC={MIN_CIRC}\n")
    f.write(f"Otsu={otsu_thresh:.1f}  dark_thresh={dark_thresh:.1f}\n\n")
    f.write(f"{'Sobel':>5} {'CnyLo':>5} {'CnyHi':>5} "
            f"{'Open':>4} {'Close':>5} {'MinA':>5} {'MaxA%':>5} "
            f"{'Holes':>5}  Details\n")
    f.write("-" * 75 + "\n")
    for row in results[:20]:
        n, st, clo, chi, ok, ck, mina, maxar, det = row
        f.write(f"{st:>5} {clo:>5} {chi:>5} "
                f"{ok:>4} {ck:>5} {mina:>5} {maxar*100:>4.0f}% "
                f"{n:>5}  {det}\n")
print("\nTop 20 results saved to best_params.txt")
