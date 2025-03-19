import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to load the saved model
def load_saved_model(model_path):
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


# Load model
model = load_saved_model('cnn_cifar10.pth')

# Train model from scratch
'''
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
'''


# Function to generate saliency maps
def generate_saliency(model, image, target_class=None):
    model.eval()
    image = image.clone()
    image.requires_grad_()

    # Forward pass
    output = model(image)

    # If target class is not provided, use the predicted class
    if target_class is None:
        target_class = output.argmax(1).item()

    # Zero gradients
    model.zero_grad()

    # Target for backprop
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0, target_class] = 1

    # Backward pass
    output.backward(gradient=one_hot_output)

    # Generate saliency map
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)

    return saliency.cpu().detach(), target_class


# Function to denormalize images
def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return tensor * std + mean


# Function to evaluate model performance
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
    except:
        precision, recall, f1 = 0, 0, 0

    # Calculate per-class accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for label, prediction in zip(all_labels, all_preds):
        if label == prediction:
            class_correct[label] += 1
        class_total[label] += 1

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_accuracy': [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    }


# Load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# Classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Evaluate and print test performance
# print("Evaluating model performance...")
# metrics = evaluate_model(model, testloader)
# print(f"Test Accuracy: {metrics['accuracy']:.4f}")
# print(f"Test Precision: {metrics['precision']:.4f}")
# print(f"Test Recall: {metrics['recall']:.4f}")
# print(f"Test F1 Score: {metrics['f1']:.4f}")
# print("\nPer-class accuracy:")
# for i, acc in enumerate(metrics['per_class_accuracy']):
#     print(f"{classes[i]}: {acc:.4f}")


def show_images_with_saliency(model, dataloader, num_images=5):
    model.eval()
    dataiter = iter(dataloader)
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

    for i in range(num_images):
        images, labels = next(dataiter)
        image = images.to(device)
        true_label = labels.item()

        # Generate saliency map
        saliency_map, pred_class = generate_saliency(model, image)

        # Denormalize
        denorm_img = denormalize(images[0])
        axes[0, i].imshow(np.transpose(denorm_img.cpu().numpy(), (1, 2, 0)))
        axes[0, i].set_title(f'True: {classes[true_label]}')
        axes[0, i].axis('off')

        # Saliency map
        axes[1, i].imshow(saliency_map.squeeze(), cmap='hot')
        pred_status = "✓" if pred_class == true_label else "✗"
        axes[1, i].set_title(f'Pred: {classes[pred_class]} {pred_status}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


# Display images and their saliency maps
show_images_with_saliency(model, testloader, num_images=5)
