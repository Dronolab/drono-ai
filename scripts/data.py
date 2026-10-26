from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class yoloDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.classes = ["Deer"] # lmao
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for img_path in sorted((self.root/"images").iterdir()):
            label_path = self.root/"labels"/f"{img_path.stem()}.txt"
            self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
    def _read_labels(self, label_path):
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    boxes.append(line.split(' '))
        return boxes
    
    def _read_images(self, image_path):
        image = None
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
        return image

