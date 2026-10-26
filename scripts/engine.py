from tqdm.auto import tqdm
import torch
def train_model(model, dataloader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    for i in tqdm(range(epochs)):
        pass
