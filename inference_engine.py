from ultralytics import YOLO
import os
engine_path = "models/best1.engine"
# Load the exported TensorRTodel
print("Loading TensorRT engine...")
tensorrt_model = YOLO(engine_path)

# Run inference on a test image (replace with your image path)
test_image = "test1.jpg"
if os.path.exists(test_image):
    print(f"Running inference on {test_image}...")
    results = tensorrt_model(test_image, imgsz=800, task="detect", conf=0.3)
    
    # Print results
    for r in results:
        print(f"Detected {len(r.boxes)} objects")
        
    # Save results
    print("Saving results...")
    os.makedirs("results", exist_ok=True)
    for i, r in enumerate(results):
        r.save(f"results/result_{i}.jpg")
    
    print("Conversion and verification complete!")
else:
    print(f"WARNING: Test image not found: {test_image}")
    print("Conversion complete. Add your own images for inference.")
