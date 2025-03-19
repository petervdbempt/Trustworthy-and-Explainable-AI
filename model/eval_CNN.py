import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision
from utils.data_utils import transformer_for_data, load_saved_model
from utils.saliency_map import show_images_with_explainability, generate_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
    model = load_saved_model('cnn_cifar10.pth')

    transform = transformer_for_data()
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # Classes in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("Evaluating model performance...")
    metrics = evaluate_model(model, testloader)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print("\nPer-class accuracy:")
    for i, acc in enumerate(metrics['per_class_accuracy']):
        print(f"{classes[i]}: {acc:.4f}")

    show_images_with_explainability(model, testloader, classes, num_images=5, use_gradCam=True)


if __name__ == '__main__':
    main()
