import time

import cv2
from hotboxDetection import get_img_hotspots

def run_detector():
    cap = cv2.VideoCapture("filtered_video.mp4") #replace with path of your video
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame for hotspot detection
        processed_frame = get_img_hotspots(200, frame, False)

        count = count + 1
        # Exit when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if cv2.waitKey(10) & 0xFF == ord('p'):
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                print("test")
                time.sleep(1)


    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows

if __name__ == '__main__':
    run_detector()
