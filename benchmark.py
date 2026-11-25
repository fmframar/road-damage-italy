"""Speed and size benchmark: how heavy and how fast is each model?

For each model we measure, on the same 50 Rome test images, one image at a time
(like the app does):
  - size: number of parameters, GFLOPs (operations per image), weights file in MB
  - time of each step of the pipeline: pre-processing, network, post-processing
  - on CPU (a cheap device without GPU) and, if available, on GPU

Usage:   python benchmark.py <runs folder>
Example: python benchmark.py /content/drive/MyDrive/road-damage/runs_kaggle

Prints one table and saves it as <runs folder>/benchmark.csv
"""
import os
import sys

import pandas as pd
import torch
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.torch_utils import get_flops

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
IMAGES_DIR = "data/yolo/images/test"
N_IMAGES = 50          # images used for timing
N_WARMUP = 5           # first runs are slower (memory allocation): not counted

# model name -> (weights file, input image size)
MODELS = {
    "YOLO11n 640": (f"{RUNS}/yolo11n_640/weights/best.pt", 640),
    "YOLO11n 1280": (f"{RUNS}/yolo11n_1280/weights/best.pt", 1280),
    "YOLO-World s (zero-shot)": ("yolov8s-worldv2.pt", 640),
}
WORLD_PROMPTS = ["pothole in the road", "crack in the asphalt", "manhole cover"]

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

images = sorted(os.listdir(IMAGES_DIR))[:N_IMAGES]
paths = [os.path.join(IMAGES_DIR, f) for f in images]


def load(name, weights):
    """Load a model. YOLO-World also needs its text prompts."""
    if "World" in name:
        model = YOLOWorld(weights)
        model.set_classes(WORLD_PROMPTS)   # text -> embeddings, computed once here
        return model
    return YOLO(weights)


rows = []
for name, (weights, imgsz) in MODELS.items():
    model = load(name, weights)

    # ---- size ----
    n_params = sum(p.numel() for p in model.model.parameters())
    gflops = get_flops(model.model, imgsz)
    file_mb = os.path.getsize(weights) / 1e6

    # ---- speed ----
    for device in DEVICES:
        for p in paths[:N_WARMUP]:                       # warm-up, not measured
            model.predict(p, imgsz=imgsz, device=device, verbose=False)

        pre, net, post = [], [], []
        for p in paths:
            result = model.predict(p, imgsz=imgsz, device=device, verbose=False)[0]
            pre.append(result.speed["preprocess"])       # milliseconds
            net.append(result.speed["inference"])
            post.append(result.speed["postprocess"])

        def mean(values):
            return sum(values) / len(values)

        total = mean(pre) + mean(net) + mean(post)
        rows.append({
            "model": name,
            "device": "GPU" if device.startswith("cuda") else "CPU",
            "params (M)": n_params / 1e6,
            "GFLOPs": gflops,
            "file (MB)": file_mb,
            "pre (ms)": mean(pre),
            "network (ms)": mean(net),
            "post (ms)": mean(post),
            "total (ms)": total,
            "images/s": 1000 / total,
        })
        print(f"{name} on {rows[-1]['device']}: {total:.1f} ms per image")

table = pd.DataFrame(rows)
print()
print(table.round(1).to_string(index=False))
table.to_csv(f"{RUNS}/benchmark.csv", index=False)
print("\nSaved to", f"{RUNS}/benchmark.csv")
