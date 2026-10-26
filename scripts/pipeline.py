import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

from models.autoencoder import AutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument("in_image", type=str, help="The path to the input image")
parser.add_argument("out_path", type=str, help="The path where the resulting image will be stored")
parser.add_argument("--yolo_params", type=str, help="The path to the yolo .pt file")
parser.add_argument("--autoencoder_params", type=str, help="The path to the autoencoder .pt file")


args = parser.parse_args()

image_path = Path(args.in_image)
assert image_path.is_file(), "in_image has to be a path to a file"
out_path = Path(args.out_path)
assert out_path.is_dir(), "out_path has to be a path to a folder"

device = "gpu" if torch.cuda.is_available() else "cpu"

yolo = None
if(args.yolo_params):
    yolo_path = Path(args.yolo_params)
    assert yolo_path.is_file(), "yolo_path has to be a path to a file"
    yolo = YOLO(yolo_path)
else:
    print(f"No yolo_path provided, defaulting to yolo11n")
    yolo_path = Path("models/pt/yolo11n")
    yolo = YOLO(yolo_path)

auto_encoder = None
if(args.autoencoder_params):
    ae_path = Path(args.autoencoder_params)
    assert ae_path.is_file(), "autoencoder_path has to be a path to a file"
    auto_encoder = AutoEncoder()
    auto_encoder.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
else:
    print(f"No autoencoder_params provided, no anomaly detection will be done")

detections = yolo(image_path)
cropped_images = []
for detection in detections:
    for box in detection.boxes.xyxy.tolist():
        xmin, ymin, xmax, ymax = box

        img = Image.open(image_path)
        cropped_images.append(F.crop(img, ymin, xmin, (ymax-ymin), (xmax-xmin)))

# for img in cropped_images:
#     img.show()    

# anomaly detection
if auto_encoder:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    anomalies = []
    for img in cropped_images:
        tensor_img = transform(img)
        anomaly = auto_encoder(tensor_img)
        transforms.ToPILImage()(anomaly).show()
        anomalies.append(anomaly)
