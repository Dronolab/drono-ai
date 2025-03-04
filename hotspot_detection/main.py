import cv2
from hotboxDetection import get_img_hotspots

def run_detector():
    cap = cv2.VideoCapture("hotbox_video.mp4")
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
        processed_frame = get_img_hotspots(240, frame, False)

        count = count + 1
        # Exit when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows

if __name__ == '__main__':
    run_detector()
