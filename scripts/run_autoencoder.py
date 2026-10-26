import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from models.autoencoder import AutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument("image_path", type=str, help="The path to the input image")
parser.add_argument("pt_file", type=str, help="The path to the autoencoder .pt file")
parser.add_argument("size", type=int, help="The size of the image that the model tales in")

args = parser.parse_args()

img_path = Path(args.image_path)
assert img_path.is_file(), "image_path has to be a path to a file"

pt_path = Path(args.pt_file)
assert pt_path.is_file(), "pt_file has to be a path to a file"

img_size = args.size

device = "gpu" if torch.cuda.is_available() else "cpu"

model = AutoEncoder(in_size=img_size)
model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))



img = Image.open(img_path)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
tensor_img = transform(img)
result = model(tensor_img)
transforms.ToPILImage()(result).show()
