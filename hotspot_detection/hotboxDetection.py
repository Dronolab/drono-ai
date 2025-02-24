"""
hotboxDetection.py

This module provides hotspot detection using openCV with threshold control
"""

import cv2
import matplotlib.pyplot as plt

def get_img_hotspots(threshold, img_path, plot=True):
    """
    Compute the hotspot's bounding boxes and display it on the image

    Parameters:
    img_path (string): Path to the img to parse.
    threshold (float): Hotspot detection threshold.(0 to 255)
    plot (boolean): True if user want to plot result 0 if not (True by default)

    Returns:
    An Array of Array of Int The coordinates and the size of the bounding boxes(x,y,height,width)
    """

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Adjust the threshold value as needed
    _, threshold = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Creating hotboxes
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bounding_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:  # Filter small regions
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x,y,w,h])
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if plot:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Hotspot Detection')
        plt.show()

    return bounding_boxes


#Testing the function
hotspots = get_img_hotspots(200,'thermal-camera-crop.jpg',True)
print(hotspots)


