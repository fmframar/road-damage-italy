import os
import random
import shutil

RAW = "data/raw/data"      # where the downloaded data is
OUT = "data/yolo"          # where we write the YOLO-ready data
NAMES = ["pothole", "crack", "manhole"]
random.seed(42)            # same shuffle every time


# ---------- Step 1: read one label file and convert it ----------
def read_label(path):
    """Each line: class + 4 corners (8 numbers). Return: class, cx, cy, w, h."""
    boxes = []
    for line in open(path):
        parts = line.split()
        if len(parts) == 0:
            continue
        cls = int(parts[0])
        nums = [float(x) for x in parts[1:]]
        xs = nums[0::2]                 # every 1st number = x of a corner
        ys = nums[1::2]                 # every 2nd number = y of a corner
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2        # center
        cy = (y_min + y_max) / 2
        w = x_max - x_min               # size
        h = y_max - y_min
        boxes.append((cls, cx, cy, w, h))
    return boxes


# ---------- Step 2: decide which "group" an image belongs to ----------
def group_of(name):
    """Images taken close in time look almost the same, so we keep
    them together (in the same split)."""
    if name.startswith("vlcsnap") and name[8:].isdigit():
        return "num_" + str(int(name[8:]) // 50)   # numbered frames: blocks of 50
    elif name.startswith("vlcsnap"):
        return name[8:22]                          # date + hour, e.g. 2025-02-18-17h
    else:
        return name[:11]                           # phone photo: date + hour


# ---------- Step 3: read everything and count ----------
image_files = sorted(os.listdir(RAW + "/images"))
print("Images:", len(image_files))

all_boxes = {}                        # image name -> list of boxes
total = [0, 0, 0]                     # count per class
for f in image_files:
    name = f.replace(".jpg", "")
    boxes = read_label(RAW + "/labels/" + name + ".txt")
    all_boxes[name] = boxes
    for b in boxes:
        total[b[0]] += 1
print("Objects per class:", dict(zip(NAMES, total)))
print("Paper says: pothole 1261, crack 2519, manhole 957")


# ---------- Step 4: put images into groups, shuffle the groups ----------
groups = {}
for name in all_boxes:
    g = group_of(name)
    if g not in groups:
        groups[g] = []
    groups[g].append(name)
print("Number of groups:", len(groups))

group_list = list(groups.keys())
random.shuffle(group_list)


# ---------- Step 5: fill train (70%), then val (15%), then test (15%) ----------
split_of = {}                          # image name -> "train"/"val"/"test"
count = 0
n = len(all_boxes)
for g in group_list:
    if count < 0.70 * n:
        split = "train"
    elif count < 0.85 * n:
        split = "val"
    else:
        split = "test"
    for name in groups[g]:
        split_of[name] = split
    count += len(groups[g])


# ---------- Step 6: write the YOLO folders ----------
if os.path.exists(OUT):
    shutil.rmtree(OUT)

for name, split in split_of.items():
    os.makedirs(OUT + "/images/" + split, exist_ok=True)
    os.makedirs(OUT + "/labels/" + split, exist_ok=True)
    shutil.copy(RAW + "/images/" + name + ".jpg", OUT + "/images/" + split + "/")
    with open(OUT + "/labels/" + split + "/" + name + ".txt", "w") as f:
        for cls, cx, cy, w, h in all_boxes[name]:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ---------- Step 7: the config file YOLO needs ----------
with open(OUT + "/data.yaml", "w") as f:
    f.write("path: " + os.path.abspath(OUT) + "\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names:\n")
    for i, name in enumerate(NAMES):
        f.write(f"  {i}: {name}\n")


# ---------- Summary ----------
for split in ["train", "val", "test"]:
    imgs = [n for n in split_of if split_of[n] == split]
    per_class = [0, 0, 0]
    for n in imgs:
        for b in all_boxes[n]:
            per_class[b[0]] += 1
    print(split, ":", len(imgs), "images", dict(zip(NAMES, per_class)))
