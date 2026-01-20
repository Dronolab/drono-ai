import torch #nécessaire pour vérifier la disponibilité de CUDA
from ultralytics import YOLO
import os
import glob



INVALID_CHOICE = "Invalid choice."

def train_script(
    model_path: str,
    data_yaml: str,
    epochs: int,
    imgsz: int = 640,
    batch: int = 16,
    device: str = 0 if torch.cuda.is_available() else 'cpu',
    workers: int = 8,
    fraction: float = 1.0,
    project: str = "runs",
    name: str = "train",
    exist_ok: bool = False,
    plots: bool = True,
    interactive: bool = True,
):
    # Check for available base models
    if(interactive):
        model_path = train_model_selector()
        data_yaml = data_selector()
        epochs = input("Enter number of epochs to train (default = 50): ").strip()
        epochs = int(epochs) if epochs else 50


    print("⚙️ Starting YOLO training with model", os.path.basename(model_path), "and dataset", os.path.basename(data_yaml))
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


    model = YOLO(model_path)

   

    try:
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device, fraction=fraction, batch=batch, amp=True, workers= workers, project=project, name=name, exist_ok=exist_ok, plots=plots)
        print("✅ Training completed.")
        print(f"📦 Trained weights saved to {project}/{name}/weights")
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
            model_path = available_models[int(choice) - 1]
            print(f"You selected: {model_path}")
            print()

            return model_path
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