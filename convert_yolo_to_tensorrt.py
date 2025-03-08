from ultralytics import YOLO
import os

# Print current working directory
print(f"Working directory: {os.getcwd()}")

# Check if model file exists
model_path = "models/best1.pt"
if os.path.exists(model_path):
    print(f"Model file found: {model_path}")
else:
    print(f"ERROR: Model file not found: {model_path}")
    exit(1)

# Load the pretrained YOLO model
print("Loading YOLO model...")
model = YOLO(model_path)

# Export the model to TensorRT format
print("Converting to TensorRT engine...")
model.export(format="engine", device=0, half=True)  # half=True for FP16 precision

# Verify the engine was created
engine_path = model_path.replace(".pt", ".engine")
if os.path.exists(engine_path):
    print(f"TensorRT engine created successfully: {engine_path}")
else:
    print("ERROR: Failed to create TensorRT engine")
    exit(1)


