import torch
import torch.nn as nn
from CNN_model import CNN
import torch.optim as optim
import torchvision
from utils.data_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transformer_for_data()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Training loop
    for epoch in range(5):  # Adjust number of epochs as needed
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}')

    # Save the trained model
    torch.save(model.state_dict(), 'cnn_cifar10.pth')

def main():
    train_model()


if __name__ == '__main__':
    main()
