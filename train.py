"""Train the YOLO variants used in the project.

Usage:   python train.py <experiment name> [output folder]
Example: python train.py yolo11n_640 /kaggle/working/runs

Every experiment changes ONE thing with respect to "yolo11n_640",
so that differences in the results can be explained by that one change.
"""
import sys

from ultralytics import YOLO

EXPERIMENTS = {
    # name:               (model file,     image size, batch, frozen layers, augmentation)
    "yolo11n_640":        ("yolo11n.pt",   640,        16,    None,          "default"),  # main model: COCO weights, full fine-tuning
    "yolo11n_1280":       ("yolo11n.pt",   1280,       8,     None,          "default"),  # bigger input: small damage = more pixels
    "yolo11n_scratch":    ("yolo11n.yaml", 640,        16,    None,          "default"),  # same network, random weights (no COCO)
    "yolo11n_frozen":     ("yolo11n.pt",   640,        16,    11,            "default"),  # COCO weights, backbone (layers 0-10) frozen
    "yolo11n_noaug":      ("yolo11n.pt",   640,        16,    None,          "none"),     # no data augmentation at all
    "yolo11n_lightaug":   ("yolo11n.pt",   640,        16,    None,          "light"),    # weaker data augmentation
    "yolo11n_strongaug":  ("yolo11n.pt",   640,        16,    None,          "strong"),   # stronger data augmentation
}

# Data augmentation = random changes applied to each training image at every epoch,
# so the network never sees exactly the same picture twice.
# Ultralytics ALREADY uses augmentation by default:
#   colour (hsv_h=0.015, hsv_s=0.7, hsv_v=0.4), shift (translate=0.1), zoom (scale=0.5),
#   left-right flip (fliplr=0.5), mosaic of 4 images (mosaic=1.0)
# Each setting below only lists the values we CHANGE; everything else stays at the default.
AUGMENTATIONS = {
    "default": {},              # change nothing

    "none": {                   # switch every augmentation off
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
    },

    "light": {                  # weaker colour, shift and zoom; no mosaic (hue and flip unchanged)
        "hsv_s": 0.35,          # colour saturation (default 0.7)
        "hsv_v": 0.2,           # brightness (default 0.4)
        "translate": 0.05,      # shift (default 0.1)
        "scale": 0.25,          # zoom (default 0.5)
        "mosaic": 0.0,          # no mosaic: the network always sees whole, real images
    },

    "strong": {                 # more variety, aimed at real driving conditions
        "hsv_v": 0.7,           # brightness: sun, shadows, cloudy days (default 0.4)
        "degrees": 5.0,         # small rotation: camera not perfectly level (default 0)
        "scale": 0.7,           # zoom: damage near and far from the car (default 0.5)
        "mixup": 0.1,           # 10% of images are blended with another image (default 0)
    },
}


def main():
    name = sys.argv[1]
    project = sys.argv[2] if len(sys.argv) > 2 else "runs"

    model_file, imgsz, batch, freeze, augmentation = EXPERIMENTS[name]
    print(f"Experiment {name}: model={model_file}, imgsz={imgsz}, batch={batch}, "
          f"freeze={freeze}, augmentation={augmentation}")

    # ".pt" file = network + COCO weights, ".yaml" file = network only (random weights)
    model = YOLO(model_file)
    model.train(
        data="data/yolo/data.yaml",  # Italian dataset (train/val/test split by groups)
        epochs=100,                  # maximum number of epochs
        patience=20,                 # stop if val mAP does not improve for 20 epochs
        imgsz=imgsz,
        batch=batch,                 # smaller batch for big images (GPU memory)
        freeze=freeze,               # None = train every layer
        device=0,                    # first GPU
        seed=0,                      # same random choices in every run
        project=project,             # parent folder of the results
        name=name,                   # results go to <project>/<name>
        exist_ok=True,               # overwrite the folder if it already exists
        # "**" unpacks the dictionary: {"hsv_v": 0.7} becomes hsv_v=0.7
        **AUGMENTATIONS[augmentation],
    )


if __name__ == "__main__":
    main()
