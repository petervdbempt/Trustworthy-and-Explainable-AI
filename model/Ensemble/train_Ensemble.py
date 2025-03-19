import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split

from utils.EarlyStopping import EarlyStopping
from utils.data_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ensemble(ensemble_size = 5):
    transform = transformer_for_data()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset, valset = train_test_split(trainset, test_size=0.2)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    for i in range(ensemble_size):
        model = CNN().to(device)
        early_stopping = EarlyStopping(patience=5, delta=0.01)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(20):  # Adjust number of epochs as needed
            train_loss = 0.0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(trainloader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(valloader)
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_state = early_stopping.load_best_model(model)
        # Save the trained model
        torch.save(best_model_state, f'cnn_cifar10_model{i}.pth')

def main():
    train_ensemble()


if __name__ == '__main__':
    main()
