import torch
import cv2
import numpy as np
import os

def one_shot_deletion_or_insertion(model, image, saliency_map, mode='deletion', threshold=0.2):
    """
    model: classification model
    image: [C, H, W] tensor
    saliency_map: [H, W] numpy array
    mode: 'insertion' or 'deletion'
    """
    device = image.device
    model.eval()

    with torch.no_grad():
        output = model(image)
        target_class = torch.argmax(output).item()
        original_conf = torch.softmax(output, dim=1)[0, target_class].item()

    if threshold is not None:
        mask = (saliency_map >= threshold).astype(float)
    else:
        mask = saliency_map

    mask = torch.tensor(mask).float().to(device)
    if image.shape[0] == 3:
        mask = mask.unsqueeze(0).repeat(3, 1, 1)

    if mode == 'deletion':
        perturbed = image * (1 - mask)  # remove everything that explanation says is important
    elif mode == 'insertion':
        baseline = torch.zeros_like(image)  # black background
        perturbed = baseline * (1 - mask) + image * mask

    with torch.no_grad():
        output = model(perturbed)
        new_conf = torch.softmax(output, dim=1)[0, target_class].item()

        if isinstance(perturbed, torch.Tensor):
            perturbed = perturbed.squeeze().detach().cpu().numpy()


        if perturbed.ndim == 3 and perturbed.shape[0] in [1, 3]:
            perturbed = np.moveaxis(perturbed, 0, -1)

            # Convert RGB to BGR
            perturbed = np.clip(perturbed, 0, 1)
            perturbed = (perturbed * 255).astype(np.uint8)
            perturbed = perturbed[:, :, ::-1]  # RGB to BGR

        gb_output_path = os.path.join('./', f'{mode}_expl.jpg')
        cv2.imwrite(gb_output_path, perturbed)

    return original_conf, new_conf