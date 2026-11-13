"""Gradio web app: upload a road image and get the detected road damage.

Run:  python app.py
It prints a public link (share=True) that works from any browser.
"""
import time

import gradio as gr
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"     # fine-tuned YOLO11n (trained on the Rome dataset)

model = YOLO(WEIGHTS)           # load the network once, when the app starts
NAMES = model.names             # {0: 'pothole', 1: 'crack', 2: 'manhole'}


def detect(image, conf):
    """image: PIL image uploaded by the user; conf: confidence threshold (0-1).
    Returns: image with boxes, table of counts per class, inference time."""
    start = time.time()
    # predict() turns the image into a tensor, runs the network,
    # and turns the output back into boxes (with NMS)
    result = model.predict(image, conf=conf, imgsz=640, verbose=False)[0]
    elapsed_ms = (time.time() - start) * 1000

    # plot() draws the boxes; it returns a BGR image (OpenCV order),
    # so we reverse the channels to get RGB for the browser
    annotated = result.plot()[:, :, ::-1]

    # count the detections of each class
    counts = {name: 0 for name in NAMES.values()}
    for cls in result.boxes.cls.tolist():
        counts[NAMES[int(cls)]] += 1
    table = [[name, n] for name, n in counts.items()]

    return annotated, table, f"{elapsed_ms:.0f} ms"


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Road image"),
        gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(label="Detected damage"),
        gr.Dataframe(headers=["class", "count"], label="Detections per class"),
        gr.Textbox(label="Inference time (pre-processing + network + post-processing)"),
    ],
    title="Road Damage Detection on Italian Roads",
    description="Upload a road image: the fine-tuned YOLO11n model finds potholes, cracks and manholes.",
)

if __name__ == "__main__":
    demo.launch(share=True)     # share=True gives a public link (needed on Colab)
