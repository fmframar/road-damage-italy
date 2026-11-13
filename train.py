"""Train the YOLO variants used in the project.

Usage:   python train.py <experiment name> [output folder]
Example: python train.py yolo11n_640 /kaggle/working/runs

Every experiment changes ONE thing with respect to "yolo11n_640",
so that differences in the results can be explained by that one change.
"""
import sys

from ultralytics import YOLO

EXPERIMENTS = {
    # name:             (model file,     image size, batch, frozen layers)
    "yolo11n_640":      ("yolo11n.pt",   640,        16,    None),  # main model: COCO weights, full fine-tuning
    "yolo11n_1280":     ("yolo11n.pt",   1280,       8,     None),  # bigger input: small damage = more pixels
    "yolo11n_scratch":  ("yolo11n.yaml", 640,        16,    None),  # same network, random weights (no COCO)
    "yolo11n_frozen":   ("yolo11n.pt",   640,        16,    11),    # COCO weights, backbone (layers 0-10) frozen
}


def main():
    name = sys.argv[1]
    project = sys.argv[2] if len(sys.argv) > 2 else "runs"

    model_file, imgsz, batch, freeze = EXPERIMENTS[name]
    print(f"Experiment {name}: model={model_file}, imgsz={imgsz}, batch={batch}, freeze={freeze}")

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
    )


if __name__ == "__main__":
    main()
