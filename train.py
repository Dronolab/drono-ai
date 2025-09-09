import torch #nécessaire pour vérifier la disponibilité de CUDA
import torchvision #nécessaire pour vérifier la disponibilité de nms  
from ultralytics import YOLO
import os
import glob
import time


device = 0 if torch.cuda.is_available() else 'cpu'
INVALID_CHOICE = "Invalid choice."

def train_script():
    # Check for available base models
    selected_model = train_model_selector()
    # Check for available datasets
    data_path = data_selector()


    print("⚙️ Starting YOLO training with model", os.path.basename(selected_model), "and dataset", os.path.basename(data_path))
    print(torch.__version__)
    print(torchvision.__version__)
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    time.sleep(2)  # Waits for 2 seconds to let user read the info

    model = YOLO(selected_model)

    epochs = input("Enter number of epochs to train (default = 50): ").strip()
    epochs = int(epochs) if epochs else 50

    try:
        model.train(data=data_path, epochs=epochs, imgsz=640, device=device)
        print("✅ Training completed.")
        print("📦 Trained weights saved to runs/detect as the latest train")
    except Exception as e:
        print(f"❌ Error during training: {e}")

def train_model_selector():
    available_models = glob.glob("models/*")

    print("Available models:")
    for idx, model_path in enumerate(available_models, start=1):
        print(f"{idx}. {model_path}")

    while(True):
        choice = input(f"Enter a choice (1-{len(available_models)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            selected_model = available_models[int(choice) - 1]
            print(f"You selected: {selected_model}")
            print()

            return selected_model
        else:
            print(INVALID_CHOICE)

def data_selector():
    yaml_files = glob.glob(os.path.join("data", "**", "*.yaml"), recursive=True)
    print("Available datasets:")
    for idx, yfile in enumerate(yaml_files, start=1):
        print(f"{idx}. {yfile}")
    
    while(True):
        choice = input(f"Enter a choice (1-{len(yaml_files)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(yaml_files):
            selected_yaml = yaml_files[int(choice) - 1]
            print(f"You selected: {selected_yaml}")
            print()

            return selected_yaml
        else:
            print(INVALID_CHOICE)