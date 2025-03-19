import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.data_utils import denormalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def show_images_with_saliency(model, dataloader, classes, num_images=5):
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