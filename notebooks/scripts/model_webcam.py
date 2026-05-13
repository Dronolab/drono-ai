import cv2 as cv
import keyboard

def run_model_on_webcam(model):
    request_exit = False

    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Couldn't find webcam")

    while True:
        if keyboard.is_pressed("q"):
            break
        
        ret, frame = cam.read()
        if not ret:
            break

        with model.inference_mode():
            results = model.train()
        



