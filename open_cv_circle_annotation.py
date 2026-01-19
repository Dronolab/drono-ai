import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

IMAGE_DIR = "data\\img"
OUTPUT_DIR = "data\\img"

MIN_AREA = 1500
MIN_CIRCULARITY = 0.6

HSV_RANGES = {
    "red target": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "yellow target": [
        (np.array([20, 160, 60]), np.array([32, 255, 230])),
    ],

    "green target": [
        (np.array([40, 80, 50]), np.array([85, 255, 255])),
    ],
    "blue target": [
        (np.array([95, 150, 40]), np.array([125, 255, 220])),
    ],
    
    # Noir : V bas, S souvent bas aussi
    "black target": [
        (np.array([0, 0, 0]), np.array([180, 255, 60])),
    ],
    # Blanc : S bas et V haut
    "white target": [
        (np.array([0, 0, 200]), np.array([180, 60, 255])),
    ],
}


def build_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (low, up) in ranges:
        mask |= cv2.inRange(hsv, low, up)
    return mask

def detect_color_circles(img, color_label):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ranges = HSV_RANGES[color_label]
    mask = build_mask(hsv, ranges)

    # Nettoyage du masque
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        # if peri <= 0:
        #     continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < MIN_CIRCULARITY:
            continue
        
       
        # x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(cnt) #bounding box 
        # ratio = w_bb / h_bb 
        # if ratio < 0.85 or ratio > 1.15: 
        #     print("box trop etiree") 
        #     continue
        
        epsilon = 0.04 * peri 
        approx = cv2.approxPolyDP(cnt, epsilon, True) 
        if len(approx) <= 4: 
            #print("polynome pas rond") 
            continue #polygone avec trop peu de coins = carré, triangle etc, pas un rond

        (x, y), r = cv2.minEnclosingCircle(cnt)
        detections.append((color_label, x, y, r))

    return detections


def write_bounding_box(filename, width, height, detections):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = IMAGE_DIR
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    for (label, x, y, r) in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        xmin = max(0, min(width - 1, int(x - r)))
        ymin = max(0, min(height - 1, int(y - r)))
        xmax = max(0, min(width - 1, int(x + r)))
        ymax = max(0, min(height - 1, int(y + r)))

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    rough = ET.tostring(annotation, "utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xml_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".xml")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(pretty)


def main():
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))]

    labels_to_detect = [
        "red target",
        "yellow target",
        "blue target",
        "green target",
        "black target",
        "white target",
    ]

    for fname in tqdm(files, desc="Pré-annotation"):
        img_path = os.path.join(IMAGE_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        all_detections = []
        for label in labels_to_detect:
            all_detections.extend(detect_color_circles(img, label))

        write_bounding_box(fname, w, h, all_detections)

if __name__ == "__main__":
    main()
