"""Zero-shot baseline: YOLO-World finds objects described with words,
WITHOUT any training on our data.

Question: do we really need fine-tuning, or can a model that already
"understands" text prompts find road damage on its own?

Rules for a fair comparison:
  1. try a few prompt sets (different words for the same 3 classes)
  2. choose the best prompt set on the VALIDATION split
  3. use the TEST split only once, with the chosen prompts

Run: python evaluate_zeroshot.py
"""
import os

from ultralytics import YOLOWorld

DATA_DIR = os.path.abspath("data/yolo")   # Italian dataset made by prepare_data.py
CLASSES = ["pothole", "crack", "manhole"]  # dataset class order (ids 0, 1, 2)

# Each prompt set describes the 3 classes, in the SAME order as CLASSES
PROMPT_SETS = {
    "simple":      ["pothole", "crack", "manhole"],
    "descriptive": ["pothole in the road", "crack in the asphalt", "manhole cover"],
    "synonyms":    ["hole in the road", "road crack", "sewer cover"],
}


def write_yaml(prompts, yaml_path):
    """Same images and labels as data/yolo, but the class names are the prompts.
    This way the validator and the model always use the same words."""
    with open(yaml_path, "w") as f:
        f.write("path: " + DATA_DIR + "\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        for i, p in enumerate(prompts):
            f.write(f"  {i}: {p}\n")


def evaluate(prompts, split):
    """Run YOLO-World with these prompts on one split. Returns the metrics."""
    yaml_path = "data/yolo_world.yaml"
    write_yaml(prompts, yaml_path)
    model = YOLOWorld("yolov8s-worldv2.pt")   # pretrained on large image-text datasets
    model.set_classes(prompts)                 # text prompts -> text embeddings (CLIP)
    return model.val(data=yaml_path, split=split, imgsz=640, verbose=False, plots=False)


def print_row(name, metrics):
    per_class = "  ".join(f"{c}={ap:.3f}" for c, ap in zip(CLASSES, metrics.box.ap50))
    print(f"{name:12s}  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}  ({per_class})")


# ---------- Step 1: choose the prompts on the validation split ----------
print("VALIDATION (to choose the prompts)")
best_name, best_map50 = None, -1.0
for name, prompts in PROMPT_SETS.items():
    m = evaluate(prompts, "val")
    print_row(name, m)
    if m.box.map50 > best_map50:
        best_name, best_map50 = name, m.box.map50
print("Best prompt set on validation:", best_name)

# ---------- Step 2: final number on the test split (used only once) ----------
print("\nTEST (final result)")
m = evaluate(PROMPT_SETS[best_name], "test")
print_row(best_name, m)
