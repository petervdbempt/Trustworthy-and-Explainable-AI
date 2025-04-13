import torch
import torchvision.transforms as transforms
from old.model.CNN_model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean

def transformer_for_data():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def load_saved_model(model_path):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

