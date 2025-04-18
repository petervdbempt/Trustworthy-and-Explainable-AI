import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from old.utils.data_utils import denormalize

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


def show_images_with_explainability(model, dataloader, classes, num_images=5, use_gradCam=True):
    model.eval()
    dataiter = iter(dataloader)
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

    for i in range(num_images):
        images, labels = next(dataiter)
        image = images.to(device)
        true_label = labels.item()

        # Generate saliency map
        if use_gradCam:
            saliency_map, pred_class = generate_gradcam(model, image, layer_name='conv1')
        else:
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


def generate_gradcam(model, image, target_class=None, layer_name='conv1'):
    model.eval()
    image = image.clone()
    image.requires_grad_()

    # Hook to capture gradients
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])  # Store gradients from the target layer

    def forward_hook(module, input, output):
        activations.append(output)  # Store feature map activations

    # Register hooks on the chosen convolutional layer
    layer = dict(model.named_modules())[layer_name]
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image)

    # Determine the target class
    if target_class is None:
        target_class = output.argmax(1).item()

    # Zero gradients
    model.zero_grad()

    # One-hot encoding for target class
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0, target_class] = 1

    # Backward pass
    output.backward(gradient=one_hot_output)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Get the gradients and activations
    gradients = gradients[0]  # Shape: [batch_size, num_channels, height, width]
    activations = activations[0]  # Shape: [batch_size, num_channels, height, width]

    # Compute weights: global average pooling of gradients
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Shape: [1, num_channels, 1, 1]

    # Compute Grad-CAM heatmap: weighted sum of activations
    gradcam = torch.sum(weights * activations, dim=1).squeeze(0)  # Shape: [height, width]

    # Apply ReLU to retain only positive contributions
    gradcam = F.relu(gradcam)

    # Normalize heatmap to [0,1] for visualization
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()

    return gradcam.cpu().detach().numpy(), target_class