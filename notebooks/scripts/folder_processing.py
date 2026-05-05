import argparse
import pathlib
from ellipse_detection import detect_ellipses
import cv2 as cv
import os
from tqdm import tqdm

def save_yolo_annotation(image_path, ellipses, output_folder):
    img = cv.imread(image_path)
    h_img, w_img = img.shape[:2]

    filename = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_folder, filename + ".txt")

    with open(label_path, "w") as f:
        for ellipse in ellipses:
            (x, y), (w, h), _ = ellipse

            # Normalisation YOLO
            x_norm = x / w_img
            y_norm = y / h_img
            w_norm = w / w_img
            h_norm = h / h_img

            class_id = 0  # We do not care about class_id

            f.write(f"{class_id} {x_norm} {y_norm} {w_norm} {h_norm}\n")

parser = argparse.ArgumentParser()

parser.add_argument("--output", help="Is the output dir", required=True)
parser.add_argument("--input", help="Is the dir where the images are at", required=True)

args = parser.parse_args()

out_path = pathlib.Path(args.output)
in_path = pathlib.Path(args.input)

assert out_path.exists(), f"Output path is not valid: {out_path}"
assert in_path.exists(), f"Input path is not valid: {in_path}"
for file in tqdm(in_path.glob("*.jpg")):
    ellipses = detect_ellipses(file)
    if len(ellipses) > 0:
        save_yolo_annotation(file, ellipses, out_path)