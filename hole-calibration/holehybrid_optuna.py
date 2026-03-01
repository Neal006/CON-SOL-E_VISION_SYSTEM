"""Optuna-based hyperparameter search for hole detection."""
import cv2
import numpy as np
import optuna
import glob
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

DARK_THRESH_RATIO = 0.7
MIN_CIRC          = 0.4

IMAGE_PATHS = sorted(glob.glob("keyhole/*.jpg"))
print(f"Pre-loading {len(IMAGE_PATHS)} images...")

preloaded = []
for path in IMAGE_PATHS:
    img     = cv2.imread(path)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img_area = gray.shape[0] * gray.shape[1]

    otsu_t, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_t = otsu_t * DARK_THRESH_RATIO
    _, dark_mask = cv2.threshold(blurred, dark_t, 255, cv2.THRESH_BINARY_INV)

    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(gx**2 + gy**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    preloaded.append({
        "name":      os.path.basename(path),
        "blurred":   blurred,
        "dark_mask": dark_mask,
        "sobel_mag": sobel_mag,
        "img_area":  img_area,
    })

print(f"Loaded {len(preloaded)} images. Starting Optuna search...")


def run_trial_on_image(entry, params):
    """Run detection pipeline on one pre-loaded image."""
    sobel_t      = params["sobel_t"]
    canny_lo     = params["canny_lo"]
    canny_hi     = params["canny_hi"]
    open_k       = params["open_k"]
    close_k      = params["close_k"]
    min_area     = params["min_area"]
    max_area     = entry["img_area"] * params["max_area_pct"]

    blurred   = entry["blurred"]
    dark_mask = entry["dark_mask"]
    sobel_mag = entry["sobel_mag"]

    _, sobel_bin = cv2.threshold(sobel_mag, sobel_t, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(blurred, canny_lo, canny_hi)

    edges = cv2.bitwise_or(sobel_bin, canny)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    interior = cv2.bitwise_and(dark_mask, cv2.bitwise_not(edges))
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    interior = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = 4 * np.pi * area / (perim ** 2)
        if circ < MIN_CIRC:
            continue
        holes.append((area, circ))
    return holes


def objective(trial):
    params = {
        "sobel_t":      trial.suggest_int("sobel_t",       10, 120),
        "canny_lo":     trial.suggest_int("canny_lo",      10,  80),
        "canny_hi":     trial.suggest_int("canny_hi",      80, 220),
        "open_k":       trial.suggest_int("open_k",         3,  13, step=2),
        "close_k":      trial.suggest_int("close_k",        5,  19, step=2),
        "min_area":     trial.suggest_int("min_area",      50, 3000),
        "max_area_pct": trial.suggest_float("max_area_pct", 0.003, 0.20),
    }

    if params["canny_lo"] >= params["canny_hi"]:
        return -999.0

    total_score = 0.0
    for entry in preloaded:
        holes = run_trial_on_image(entry, params)
        if holes:
            avg_circ = sum(c for _, c in holes) / len(holes)
            bonus = 0.15 if len(holes) == 1 else 0.0
            total_score += avg_circ + bonus
        else:
            total_score -= 0.5

    return total_score / len(preloaded)


N_TRIALS = 200
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best     = study.best_params
best_val = study.best_value

print(f"\n{'='*55}")
print(f"BEST SCORE : {best_val:.4f}")
print(f"{'='*55}")
for k, v in best.items():
    print(f"  {k:>15} = {v}")

with open("optuna_best_params.txt", "w") as f:
    f.write(f"Best score: {best_val:.4f}\n\n")
    for k, v in best.items():
        f.write(f"{k} = {v}\n")
print("\nSaved to optuna_best_params.txt")

print(f"\n{'─'*55}")
print("Validation — best params on all images:")
print(f"{'─'*55}")
total_holes = 0
images_hit  = 0
for entry in preloaded:
    holes = run_trial_on_image(entry, best)
    detail = "  ".join(f"[a={int(a)} c={c:.2f}]" for a, c in holes) if holes else "NO HOLES"
    print(f"  {entry['name']:<22} {len(holes):>2} hole(s)  {detail}")
    total_holes += len(holes)
    if holes:
        images_hit += 1

print(f"\nImages with holes : {images_hit}/{len(preloaded)}")
print(f"Total holes found : {total_holes}")
