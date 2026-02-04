import os
from requests import get
from ultralytics import YOLO
import cv2
import numpy as np
   

# ÉTAPE 3: output le unit vector   
def uni_vector_from_pixel(u, v):
    K = np.array([[1089.55484, 0.0, 988.161623],
                [0.0, 1089.07737, 553.229288],
                [0.0, 0.0, 1.0]])
    
    
    def get_unit_vector(u, v, k):
        k_inv = np.linalg.inv(k)

        pixel_point = np.array([u, v, 1.0])
        p_3d = k_inv.dot(pixel_point)
        unit_vector = p_3d / np.linalg.norm(p_3d)
        return unit_vector

    vector = get_unit_vector(u, v, K)
    return vector


#Étape 2: undistorted l'image
def undistort_image(u,v):
    if not os.path.exists('siyi_a8_calib.npz'):
        print("Error: siyi_a8_calib.npz not found. Run the calibration script first.")
        exit()

    data = np.load('siyi_a8_calib.npz')
    mtx = data['mtx']
    dist = data['dist']
    h = 1080
    w = 1920

    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    src_point = np.array([[[u, v]]], dtype=np.float32)

    undistorted_point = cv2.undistortPoints(src_point, mtx, dist, None, new_camera_mtx)

    undistorted_x, undistorted_y = undistorted_point[0][0]

    return uni_vector_from_pixel(undistorted_x, undistorted_y)


#Étape 1 obtenir les coordonnées XY
def predict_source(conf=0.5):
    try:
        model_path = "./yolo11n.pt"
        model = YOLO(model_path)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            print("Error: Could not open webcam.", flush=True)

            return
        
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1920, 1080))

            if not ret:
                print("Error: Could not read frame from webcam.", flush=True)

                break

            result = model.predict(
                source=frame,
                conf=conf,
                show=False,
                verbose=False,
                save=False,
                imgsz=640)

            if(result[0].boxes is not None and len(result[0].boxes) > 0):
                names = result[0].names
                class_ids = result[0].boxes.cls.tolist()
                cx, cy, _, _ = result[0].boxes.xywh[0].tolist()

                #nom de l'objet detecte + coordonnes centrale du bounding box + vecteur unitaire calcule
                print(names[class_ids[0]]+ " ", flush=True)
                print(str(cx) +" " + str(cy), flush=True)
                print(undistort_image(int(cx), int(cy)), flush=True)
                

                # this part will be removed when break is removed
                annotated_frame = result[0].plot()
                cv2.imwrite("./webcam_frame_pred.png", annotated_frame)




            #this break will be removed later
            break

        cap.release()
        cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)


if __name__ == "__main__":
    predict_source()
