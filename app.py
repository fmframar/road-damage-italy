"""Gradio web app: upload a road image and get the detected road damage.

Run:  python app.py
It prints a public link (share=True) that works from any browser.

Needs the trained weights in weights/ and (optionally) example images in examples/.
"""
import os

import gradio as gr
import torch
from ultralytics import YOLO

# Two fine-tuned YOLO11n models (same network, different input size):
# name shown in the app -> (weights file, image size used in training)
MODEL_FILES = {
    "YOLO11n 640 - fast": ("weights/yolo11n_640.pt", 640),
    "YOLO11n 1280 - more accurate": ("weights/yolo11n_1280.pt", 1280),
}
DEFAULT_MODEL = "YOLO11n 640 - fast"

# Load every network once, when the app starts (not at every click)
MODELS = {name: (YOLO(path), imgsz) for name, (path, imgsz) in MODEL_FILES.items()}
DEVICE = "GPU" if torch.cuda.is_available() else "CPU"

# Example images from the Rome TEST set (never seen during training)
EXAMPLE_FILES = [
    "examples/shadow.jpg",          # strong building shadow on the road
    "examples/manholes.jpg",        # two manholes and a crack
    "examples/pothole_cracks.jpg",  # pothole and several cracks
    "examples/manhole_cover.jpg",   # manhole-like cover next to a crack
]
EXAMPLES = [f for f in EXAMPLE_FILES if os.path.exists(f)]   # skip missing files


def detect(image, model_name, conf):
    """image: PIL image from the user; model_name: key of MODELS;
    conf: confidence threshold (0-1).
    Returns: image with boxes, table of counts per class, timing text."""
    if image is None:
        return None, [], "Please upload an image first."

    model, imgsz = MODELS[model_name]

    # predict() does the whole pipeline:
    #   image -> resized tensor [1, 3, H, W] with values in [0, 1]
    #   -> network -> raw boxes -> NMS (removes duplicate boxes) -> final boxes
    result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]

    # plot() draws the boxes; it returns a BGR image (OpenCV order),
    # so we reverse the channels to get RGB for the browser
    annotated = result.plot()[:, :, ::-1]

    # count the detections of each class
    counts = {name: 0 for name in model.names.values()}
    for cls in result.boxes.cls.tolist():
        counts[model.names[int(cls)]] += 1
    table = [[name, n] for name, n in counts.items()]

    # time of each step of the pipeline, in milliseconds
    s = result.speed
    total = s["preprocess"] + s["inference"] + s["postprocess"]
    timing = (f"{total:.0f} ms on {DEVICE} = "
              f"{s['preprocess']:.0f} pre-processing + "
              f"{s['inference']:.0f} network + "
              f"{s['postprocess']:.0f} post-processing")

    return annotated, table, timing


# ---------- Page layout ----------
with gr.Blocks(title="Road Damage Detection") as demo:
    gr.Markdown(
        "# Road Damage Detection on Italian Roads\n"
        "Upload a road image (or click an example below): a YOLO11n network, "
        "pretrained on COCO and fine-tuned on Italian roads, finds "
        "**potholes**, **cracks** and **manholes**."
    )

    with gr.Row():
        with gr.Column():                      # left: inputs
            image_in = gr.Image(type="pil", label="Road image")
            model_in = gr.Radio(list(MODELS.keys()), value=DEFAULT_MODEL, label="Model")
            conf_in = gr.Slider(0.05, 0.90, value=0.25, step=0.05,
                                label="Confidence threshold")
            button = gr.Button("Detect damage", variant="primary")

        with gr.Column():                      # right: outputs
            image_out = gr.Image(label="Detected damage")
            table_out = gr.Dataframe(headers=["class", "count"],
                                     label="Detections per class")
            time_out = gr.Textbox(label="Inference time")

    if EXAMPLES:
        gr.Examples(examples=[[f, DEFAULT_MODEL, 0.25] for f in EXAMPLES],
                    inputs=[image_in, model_in, conf_in],
                    label="Examples from the Rome test set")

    button.click(fn=detect,
                 inputs=[image_in, model_in, conf_in],
                 outputs=[image_out, table_out, time_out])

if __name__ == "__main__":
    demo.launch(share=True)     # share=True gives a public link (needed on Colab)
