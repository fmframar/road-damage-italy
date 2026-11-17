"""Make the analysis figures for the README and the slides.

  1. figures/classical_baseline.png
     Why classical image processing is unreliable: thresholding and Canny
     edges react to shadows, manholes and road markings, not only to damage.
  2. figures/rome_predictions.png
     True boxes vs model predictions on Rome test images.
  3. figures/czech_predictions.png
     Same, on Czech images: is the drop a real failure of the model,
     or do the two datasets simply draw the boxes differently?

Usage:   python make_figures.py <weights> [image size]
Example: python make_figures.py runs_kaggle/yolo11n_640/weights/best.pt 640
"""
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use("Agg")                     # save figures to files, no window
import matplotlib.pyplot as plt

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "weights/best.pt"
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 640
OUT = "figures"
NAMES = ["pothole", "crack", "manhole"]
GREEN = (0, 200, 0)                        # true boxes (RGB)
RED = (255, 0, 0)                          # predicted boxes (RGB)

os.makedirs(OUT, exist_ok=True)


# ---------- small helpers ----------
def load_rgb(path):
    """Read an image file as RGB (OpenCV reads BGR)."""
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def label_path_of(img_dir, file_name):
    """data/.../images/test/x.jpg -> data/.../labels/test/x.txt"""
    name = os.path.splitext(file_name)[0] + ".txt"
    return os.path.join(img_dir.replace("images", "labels"), name)


def classes_in(label_path):
    """Set of class ids present in a YOLO label file."""
    return {int(line.split()[0]) for line in open(label_path) if line.strip()}


def read_labels(label_path, w, h):
    """YOLO label file -> list of (class, x1, y1, x2, y2) in pixels."""
    boxes = []
    for line in open(label_path):
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()
        cx, cy = float(cx) * w, float(cy) * h      # normalized -> pixels
        bw, bh = float(bw) * w, float(bh) * h
        boxes.append((int(cls), int(cx - bw / 2), int(cy - bh / 2),
                      int(cx + bw / 2), int(cy + bh / 2)))
    return boxes


def draw(img, boxes, color, with_conf=False):
    """Draw boxes (class, x1, y1, x2, y2[, confidence]) on an RGB image."""
    for b in boxes:
        cls, x1, y1, x2, y2 = b[:5]
        text = f"{NAMES[cls]} {b[5]:.2f}" if with_conf else NAMES[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ---------- Figure 1: classical image processing ----------
def classical_figure():
    img_dir = "data/yolo/images/test"
    files = sorted(os.listdir(img_dir))
    # images with BOTH a pothole (0) and a manhole (2): the hardest case
    chosen = [f for f in files if {0, 2} <= classes_in(label_path_of(img_dir, f))][:3]

    fig, axes = plt.subplots(len(chosen), 3, figsize=(18, 4 * len(chosen)))
    for row, f in zip(axes, chosen):
        rgb = load_rgb(os.path.join(img_dir, f))
        h, w = rgb.shape[:2]
        true = read_labels(label_path_of(img_dir, f), w, h)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Thresholding: keep the dark pixels (Otsu chooses the threshold)
        _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Canny: keep the strong edges (after a small blur to remove noise)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

        panels = [rgb.copy(),
                  cv2.cvtColor(dark, cv2.COLOR_GRAY2RGB),
                  cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)]
        titles = ["Image + true boxes", "Thresholding (dark pixels)", "Canny edges"]
        for ax, panel, title in zip(row, panels, titles):
            draw(panel, true, GREEN)            # true boxes on every panel
            ax.imshow(panel)
            ax.set_title(title)
            ax.axis("off")

    fig.suptitle("Classical image processing: shadows, manholes and markings "
                 "give the same response as damage", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/classical_baseline.png", dpi=110)
    plt.close(fig)
    print("Saved", f"{OUT}/classical_baseline.png")


# ---------- Figures 2 and 3: true boxes vs predictions ----------
def prediction_figure(model, img_dir, out_name, title, n=6, seed=0):
    files = sorted(os.listdir(img_dir))
    # only images that contain at least one true box
    files = [f for f in files if classes_in(label_path_of(img_dir, f))]
    random.seed(seed)                       # same images every time
    chosen = random.sample(files, n)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, f in zip(axes.flat, chosen):
        path = os.path.join(img_dir, f)
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        true = read_labels(label_path_of(img_dir, f), w, h)

        result = model.predict(path, conf=0.25, imgsz=IMGSZ, verbose=False)[0]
        pred = []
        for xyxy, cls, conf in zip(result.boxes.xyxy.tolist(),
                                   result.boxes.cls.tolist(),
                                   result.boxes.conf.tolist()):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            pred.append((int(cls), x1, y1, x2, y2, conf))

        draw(rgb, true, GREEN)
        draw(rgb, pred, RED, with_conf=True)
        ax.imshow(rgb)
        ax.set_title(f"{f}  (true: {len(true)}, predicted: {len(pred)})")
        ax.axis("off")

    fig.suptitle(f"{title} - green: true boxes, red: predictions (conf >= 0.25)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{out_name}", dpi=110)
    plt.close(fig)
    print("Saved", f"{OUT}/{out_name}")


if __name__ == "__main__":
    classical_figure()

    from ultralytics import YOLO             # imported here: figure 1 does not need it
    model = YOLO(WEIGHTS)
    prediction_figure(model, "data/yolo/images/test", "rome_predictions.png",
                      "Rome test set")
    prediction_figure(model, "data/rdd_czech_yolo/images/test", "czech_predictions.png",
                      "Czech images (domain shift)")
