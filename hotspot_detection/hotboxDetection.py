"""
hotboxDetection.py

This module provides hotspot detection using openCV with threshold control
"""

import cv2
import matplotlib.pyplot as plt

def merge_close_boxes(boxes, distance_threshold=20):
    merged = []
    while boxes:
        base = boxes.pop(0)
        x, y, w, h = base
        changed = True
        while changed:
            changed = False
            for i, (x2, y2, w2, h2) in enumerate(boxes):
                # Check if boxes are close (extend bounding box by threshold)
                if not (x + w + distance_threshold < x2 or x2 + w2 + distance_threshold < x or
                        y + h + distance_threshold < y2 or y2 + h2 + distance_threshold < y):
                    # Merge the boxes
                    x_new = min(x, x2)
                    y_new = min(y, y2)
                    w_new = max(x + w, x2 + w2) - x_new
                    h_new = max(y + h, y2 + h2) - y_new
                    base = [x_new, y_new, w_new, h_new]
                    boxes.pop(i)
                    x, y, w, h = base
                    changed = True
                    break
        merged.append(base)
    return merged


def get_img_hotspots(threshold, frame, plot=True, distance_threshold=20):
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    _, thresh_img = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, w, h])

    merged_boxes = merge_close_boxes(bounding_boxes, distance_threshold)

    for x, y, w, h in merged_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Thermal Hotspot Detection", frame)
    if plot:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Hotspot Detection (Merged)')
        plt.show()

    return merged_boxes
