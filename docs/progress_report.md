# Road Damage Detection on Italian Roads: progress report

**Repository:** https://github.com/fmframar/road-damage-italy

## 1. Problem

Detect **potholes, cracks and manholes** in images from a low-cost dashboard camera.

Classical image processing is unreliable in real driving conditions. Thresholding marks shadows, vegetation and the area under cars as "dark" like a pothole. Canny edges respond equally to lane markings, cars and manholes. A real crack can give no response at all (see `figures/classical_baseline.png`).

## 2. Data

- **Rome dataset** (Giordani et al., *Scientific Reports* 16:15318, 2026; CC BY 4.0): 2009 images (640×360), 1261 potholes, 2519 cracks, 957 manholes.
  - 1907 frames from a GoPro on a moving car, 102 smartphone photos.
- **Split by groups, to avoid data leakage.** Video frames taken a few seconds apart are almost identical. Images are grouped (GoPro: 5-minute windows; phone photos: whole day) and each group goes entirely into one split.

| Split | Images | Pothole | Crack | Manhole |
|---|---|---|---|---|
| train | 1440 | 858 | 1740 | 706 |
| val | 277 | 195 | 328 | 140 |
| test | 292 | 208 | 451 | 111 |

- **Out-of-domain test:** RDD2022 Czech (Arya et al., 2022; CC BY-SA 4.0), 2829 images. Three crack types are merged into "crack"; there are no manhole labels, so only pothole and crack are evaluated.

## 3. Models

- **YOLO11n** (Ultralytics), pretrained on **COCO**, fine-tuned on the Rome train set.
- 4 variants, each changing **one** thing (trained on Kaggle, T4 GPU):

| Variant | What changes |
|---|---|
| `yolo11n_640` | reference: COCO weights, full fine-tuning, 640 px input |
| `yolo11n_1280` | 1280 px input (small damage = more pixels) |
| `yolo11n_frozen` | backbone frozen: only neck and head are trained |
| `yolo11n_scratch` | same network, random weights (no COCO) |

- **Zero-shot baseline:** YOLO-World v2 (open-vocabulary, CLIP text encoder), no training on our data. Prompts chosen on the validation split.

## 4. Results on the Rome test set (AP50)

| Model | Pothole | Crack | Manhole | **mAP50** | mAP50-95 |
|---|---|---|---|---|---|
| YOLO-World zero-shot | 0.002 | 0.000 | 0.151 | 0.051 | 0.016 |
| YOLO11n from scratch | 0.163 | 0.081 | 0.389 | 0.211 | 0.095 |
| YOLO11n frozen backbone | 0.276 | 0.204 | 0.536 | 0.339 | 0.156 |
| YOLO11n 640 | 0.220 | 0.209 | 0.617 | 0.349 | 0.152 |
| **YOLO11n 1280** | 0.260 | 0.221 | 0.646 | **0.376** | **0.161** |

**Findings**
- **Fine-tuning is needed:** the zero-shot model is about 7× worse and 5× bigger.
- **COCO pretraining matters:** from scratch loses about 40% of mAP50 on the test set. The validation set suggested only about 14%.
- **Higher resolution helps:** 1280 px gives about +8% mAP50, mostly on potholes and manholes.
- **Comparison with the dataset paper:** YOLO11n reaches 0.454 mAP50 on validation in the paper (random split, 60 hyperparameter-tuning runs). Ours reaches 0.429 on validation (group split, default hyperparameters).

## 5. Robustness to another country (Rome → Czech)

| Model | Rome | Czech | Drop |
|---|---|---|---|
| YOLO11n 640 | 0.214 | 0.042 | −80% |
| YOLO11n 1280 | 0.241 | 0.026 | −89% |
| Frozen backbone | 0.240 | 0.021 | −91% |
| From scratch | 0.122 | 0.013 | −90% |

*mAP50 over the 2 shared classes (pothole, crack).*

Visual analysis (`figures/czech_predictions.png`) shows **two causes**:
1. **Different annotation style:** the model finds the damaged area but draws one large box where RDD has several small boxes, so the metric counts it as errors.
2. **Real domain shift:** thin isolated cracks are missed completely (Rome annotates cracks as damaged areas), and road arrows are mistaken for damage.

## 6. Speed and size (one image at a time, Colab)

| Model | Params | GFLOPs | Weights | CPU | GPU |
|---|---|---|---|---|---|
| **YOLO11n 640** | 2.6 M | 6.6 | 5.5 MB | **81 ms (12 img/s)** | 13 ms |
| YOLO11n 1280 | 2.6 M | 27.0 | 5.6 MB | 309 ms (3 img/s) | 13 ms |
| YOLO-World s | 12.8 M | 32.1 | 25.9 MB | 272 ms (4 img/s) | 11 ms |

- On CPU, 1280 is 3.8× slower than 640, matching about 4× more operations. The weights are the same size: resolution changes the computation, not the network.
- At 50 km/h, 640 on CPU analyses one image every 1.1 m of road.

## 7. Application

Gradio web app on Google Colab:
- upload an image or click one of 4 examples from the Rome test set;
- choose the model (640 fast / 1280 more accurate) and the confidence threshold;
- output: image with boxes, count per class, time of each pipeline step (pre-processing, network, post-processing) on CPU or GPU.

## 8. Next steps

- README with theory (YOLO architecture, tensor pipeline), results and figures.
- Slides and video.
- Possible extra: domain adaptation on part of the Czech data, comparing Czech-only fine-tuning with Czech + Rome, to measure forgetting.
