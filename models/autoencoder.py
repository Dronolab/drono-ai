import torch.nn as nn
import torchvision.transforms as transforms;
class AutoEncoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Linear(..., 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, ...),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))