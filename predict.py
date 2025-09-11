from glob import glob
from ultralytics import YOLO
import cv2

from train import train_script

def predict_model_selector():
    while(True):
        choice = input("Enter choice 1 for default or 2 for trained: ").strip()
        if choice == "1":
            return "models/yolo11n.pt"  # Default model
    
        elif choice == "2":
            available_models = glob("runs/detect/*/weights/*.pt")
            print("Available models:")
            for idx, model_path in enumerate(available_models, start=1):
                print(f"{idx}. {model_path}")
            choice = input(f"Enter a choice (1-{len(available_models)}) default is 3: ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                selected_model = available_models[int(choice) - 1]
                print()

                return selected_model

def predict_source(source, conf=0.5, show=False):
    """
    Perform YOLO predictions on the specified source (image, video, or webcam).

    Args:
        source (str or int): Path to an image or video file, or 0 for webcam.
        model_path (str): Path to the YOLO model file.
        conf (float): Confidence threshold for predictions.
        show (bool): Whether to display the predictions.
    """
    print("Would you like to use a default model or a trained one?")
    model_path = predict_model_selector()

    # Load the YOLO model
    model = YOLO(model_path)

    # Check if the source is a webcam (integer 0 or other)
    if isinstance(source, int) or source == "webcam":
        cap = cv2.VideoCapture(0 if source == "webcam" else source)
        if not cap.isOpened():
            print("Error: Could not open webcam.")

            return
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from webcam.")
    
                break
            # Run inference on each frame
            result = model.predict(source=frame, conf=conf, show=show)

            annotated_frame = result[0].plot()  # Ultralytics returns a list of results
            cv2.imshow("Webcam Predictions", annotated_frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # For image or video files
        model.predict(source=source, conf=conf, save=True, show=show)
        print(f"Predictions completed for: {source}")

if __name__ == "__main__":
    print("Select an input type:")
    print("1. Webcam")
    print("2. Image")
    print("3. Video")
    print("4. Training")
    choice = input("Enter choice (1/2/3/4): ").strip()
    print()

    if choice == "1":
        predict_source(source="webcam")
    elif choice == "2":
        image_path = input("Enter the path to the image: ").strip()
        predict_source(source=image_path)
    elif choice == "3":
        video_path = input("Enter the path to the video: ").strip()
        predict_source(source=video_path)
    elif choice == "4":
        train_script()
    else:
        print("Invalid choice. Please run the script again.")
