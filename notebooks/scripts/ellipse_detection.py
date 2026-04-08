from PIL import Image
import cv2 as cv
import numpy as np

def detect_ellipses(image_path: str):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv.GaussianBlur(gray, (7, 7), 1.5)

    # Edge detection
    edges = cv.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    mask = np.zeros_like(gray)

    ellipses = []
    for cnt in contours:
        # Need at least 5 points to fit ellipse
        if len(cnt) < 5:
            continue

        ellipse = cv.fitEllipse(cnt)

        (x, y), (MA, ma), angle = ellipse

        area = cv.contourArea(cnt)
        if area < 300:  # tweak this
            continue

        ratio = max(MA, ma) / min(MA, ma)
        if ratio > 5:  # changer pour accepter des plus ellipses plus extremes
            continue
        
        ellipses.append(ellipse)
    return ellipses
