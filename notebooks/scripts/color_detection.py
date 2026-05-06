from PIL import Image
import cv2 as cv
from pathlib import Path
import numpy as np
    
def get_color_label_hsv(img, x, y):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    x, y = int(x), int(y)
    patch = hsv[max(0,y-2):y+3, max(0,x-2):x+3]

    h = np.mean(patch[:,:,0])
    s = np.mean(patch[:,:,1])
    v = np.mean(patch[:,:,2])


    # ⚫ noir
    if v < 50:
        return "black_target"

    # ⚪ blanc
    if v > 200 and s < 40:
        return "white_target"

    # 🔴 rouge (wrap autour de 0)
    if (h < 10 or h > 170) and s > 100:
        return "red_target"

    # 🟡 jaune
    if 20 < h < 35 and s > 100:
        return "yellow_target"

    # 🟢 vert
    if 40 < h < 85 and s > 100:
        return "green_target"

    # 🔵 bleu
    if 90 < h < 130 and s > 100:
        return "blue_target"

    return "unknown"
