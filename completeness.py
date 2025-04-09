import torch

def one_shot_deletion_or_insertion(model, image, saliency_map, mode='deletion', threshold=None):
    """
    model: classification model
    image: [C, H, W] tensor
    saliency_map: [H, W] numpy array
    mode: 'insertion' or 'deletion'
    """
    device = image.device
    model.eval()

    with torch.no_grad():
        output = model(image) #.unsqueeze(0)
        target_class = torch.argmax(output).item()
        original_conf = torch.softmax(output, dim=1)[0, target_class].item()

    # Optionally binarize the map if threshold is given
    if threshold is not None:
        mask = (saliency_map >= threshold).astype(float)
    else:
        mask = saliency_map  # assume continuous map (already normalized between 0–1)

    mask = torch.tensor(mask).float().to(device)
    if image.shape[0] == 3:
        mask = mask.unsqueeze(0).repeat(3, 1, 1)

    # Apply perturbation
    if mode == 'deletion':
        perturbed = image * (1 - mask)  # remove everything that explanation says is important
    elif mode == 'insertion':
        baseline = torch.zeros_like(image)  # black background
        perturbed = baseline * (1 - mask) + image * mask
    else:
        raise ValueError("mode must be 'insertion' or 'deletion'")

    with torch.no_grad():
        output = model(perturbed)#.unsqueeze(0)
        new_conf = torch.softmax(output, dim=1)[0, target_class].item()

    return original_conf, new_conf