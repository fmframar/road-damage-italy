"""Convert the Czech part of RDD2022 (Pascal VOC XML) to YOLO format.

We use it ONLY as an out-of-domain test set: the model is trained on Rome
and then evaluated on Czech roads. RDD has no manhole labels, so only
the two shared classes (pothole, crack) are evaluated.
"""
import os
import shutil
import xml.etree.ElementTree as ET   # standard library XML reader

from PIL import Image                # installed together with ultralytics

RAW = "data/rdd_czech/Czech/train"   # only "train" has labels
OUT = "data/rdd_czech_yolo"          # where we write the YOLO-ready data
NAMES = ["pothole", "crack", "manhole"]   # SAME order as the Italian dataset

# RDD class name -> our class id
CLASS_MAP = {
    "D00": 1,   # longitudinal crack -> crack
    "D10": 1,   # transverse crack   -> crack
    "D20": 1,   # alligator crack    -> crack
    "D40": 0,   # pothole            -> pothole
}


# ---------- Step 1: read one XML file and convert its boxes ----------
def read_xml(xml_path, img_w, img_h):
    """Return a list of (class, cx, cy, w, h) normalized to [0, 1],
    plus the RDD names we read (to count them)."""
    boxes = []
    raw_names = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        raw_names.append(name)
        if name not in CLASS_MAP:        # unknown class: skip it
            continue
        bb = obj.find("bndbox")
        x_min = float(bb.find("xmin").text)
        y_min = float(bb.find("ymin").text)
        x_max = float(bb.find("xmax").text)
        y_max = float(bb.find("ymax").text)
        cx = (x_min + x_max) / 2 / img_w  # center, divided by image size
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w       # size, divided by image size
        h = (y_max - y_min) / img_h
        boxes.append((CLASS_MAP[name], cx, cy, w, h))
    return boxes, raw_names


# ---------- Step 2: prepare the output folders ----------
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT + "/images/test")
os.makedirs(OUT + "/labels/test")


# ---------- Step 3: convert every image ----------
image_files = sorted(os.listdir(RAW + "/images"))
print("Images:", len(image_files))

raw_count = {}                 # RDD name -> how many boxes
total = [0, 0, 0]              # our classes -> how many boxes
empty = 0                      # images with no damage at all
for f in image_files:
    name = os.path.splitext(f)[0]
    img_path = RAW + "/images/" + f
    img_w, img_h = Image.open(img_path).size   # real size, read from the image

    boxes, raw_names = read_xml(RAW + "/annotations/xmls/" + name + ".xml", img_w, img_h)
    for n in raw_names:
        raw_count[n] = raw_count.get(n, 0) + 1
    for b in boxes:
        total[b[0]] += 1
    if len(boxes) == 0:
        empty += 1

    shutil.copy(img_path, OUT + "/images/test/")
    with open(OUT + "/labels/test/" + name + ".txt", "w") as out:
        for cls, cx, cy, w, h in boxes:
            out.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

print("RDD names found:", dict(sorted(raw_count.items())))
print("Paper says: D00 988, D10 399, D20 161, D40 197")
print("After mapping:", dict(zip(NAMES, total)))
print("Images without damage:", empty)


# ---------- Step 4: the config file YOLO needs ----------
# YOLO always wants a "val" entry: we point it to the same test images.
with open(OUT + "/data.yaml", "w") as f:
    f.write("path: " + os.path.abspath(OUT) + "\n")
    f.write("train: images/test\n")
    f.write("val: images/test\n")
    f.write("test: images/test\n")
    f.write("names:\n")
    for i, name in enumerate(NAMES):
        f.write(f"  {i}: {name}\n")
print("Done. Data in", OUT)
