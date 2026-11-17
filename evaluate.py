"""Evaluate every trained model on the two TEST sets:
  - rome:  test split of the Italian dataset (same domain as training)
  - czech: RDD2022 Czech images (different country = domain shift)

Usage:   python evaluate.py <runs folder>
Example: python evaluate.py /content/drive/MyDrive/road-damage/kaggle/runs

Prints one table and saves it as <runs folder>/test_results.csv
"""
import sys

import pandas as pd
from ultralytics import YOLO

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
CLASSES = ["pothole", "crack", "manhole"]

# model name -> image size used during training (we test with the same size)
MODELS = {
    "yolo11n_640": 640,
    "yolo11n_1280": 1280,
    "yolo11n_frozen": 640,
    "yolo11n_scratch": 640,
}

# test set name -> dataset description file
TEST_SETS = {
    "rome": "data/yolo/data.yaml",
    "czech": "data/rdd_czech_yolo/data.yaml",
}

rows = []
for model_name, imgsz in MODELS.items():
    model = YOLO(f"{RUNS}/{model_name}/weights/best.pt")   # best epoch on validation

    for test_name, data in TEST_SETS.items():
        print(f"Evaluating {model_name} on {test_name} ...")
        m = model.val(data=data, split="test", imgsz=imgsz, verbose=False, plots=False)

        row = {"model": model_name, "test": test_name}
        # AP50 of each class that has labels in this test set
        # (Czech has no manhole labels, so manhole is missing there)
        for cls_index, ap in zip(m.box.ap_class_index, m.box.ap50):
            row[CLASSES[cls_index]] = ap
        row["mAP50-95"] = m.box.map
        rows.append(row)

table = pd.DataFrame(rows)

# Fair comparison between Rome and Czech: mean AP50 over the 2 shared classes only
table["mAP50 (pothole+crack)"] = table[["pothole", "crack"]].mean(axis=1)

print()
print(table.round(3).to_string(index=False))
table.to_csv(f"{RUNS}/test_results.csv", index=False)
print("\nSaved to", f"{RUNS}/test_results.csv")
