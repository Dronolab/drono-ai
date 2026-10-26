from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeerDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        image_size=(128, 128),
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / split / "images"
        self.label_dir = self.data_dir / split / "labels"

        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.samples = []

        image_extensions = {".jpg", ".jpeg", ".png"}

        for image_path in self.image_dir.iterdir():

            if image_path.suffix.lower() not in image_extensions:
                continue

            label_path = self.label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            with open(label_path, "r") as file:
                for line in file:
                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    class_id, x_center, y_center, width, height = map(
                        float,
                        values,
                    )

                    self.samples.append(
                        (
                            image_path,
                            x_center,
                            y_center,
                            width,
                            height,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        (
            image_path,
            x_center,
            y_center,
            width,
            height,
        ) = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        image_width, image_height = image.size

        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        deer = image.crop((x1, y1, x2, y2))

        if self.transform:
            deer = self.transform(deer)

        return deer