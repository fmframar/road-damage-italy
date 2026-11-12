# road-damage-italy

## 3. Dataset

**Source.** *Road Damage Dataset: Potholes, Cracks, and Manholes*
(Giordani et al., Scientific Reports, 2026), license CC BY 4.0,
[Zenodo](https://doi.org/10.5281/zenodo.17834373).

- 2009 images (640×360) from Rome and nearby towns (e.g. Sacrofano)
- 1907 video frames from a GoPro HERO7 mounted on a moving vehicle
- 102 photos from a Samsung Galaxy A14 (static captures)
- 3 classes: `0 pothole` (1261), `1 crack` (2519), `2 manhole` (957)

**Label conversion.** Each label gives the 4 corners of the box
(8 normalized numbers). Boxes are axis-aligned, so `prepare_data.py`
converts them to YOLO format (center x, center y, width, height)
using the min/max of the corners.

**Train/val/test split without data leakage.** Consecutive frames of a
video look almost identical. With a random split, the same pothole could
appear in both train and test, and the test mAP would be inflated.
We therefore split by *groups* of images taken close in time, and each
group goes entirely into one split:

| Image type | Example name | Group |
|---|---|---|
| GoPro frame with timestamp | `vlcsnap-2025-02-19-14h00m39s870` | 5-minute window (vehicle moving → different road) |
| GoPro numbered frame | `vlcsnap-00081` | block of 20 consecutive numbers |
| Phone photo | `20250216_164325` | whole day (taken on foot, same damage may be photographed more than once) |

Groups are shuffled with a fixed seed (42) and assigned to
train (~70%), val (~15%) and test (~15%).

| Split | Images | Pothole | Crack | Manhole |
|---|---|---|---|---|
| train | | | | |
| val | | | | |
| test | | | | |
