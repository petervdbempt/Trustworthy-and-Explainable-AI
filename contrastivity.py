import argparse
import os
import random

import cv2
import numpy as np
import torch
from sklearn.metrics import jaccard_score
from torchvision import models
from torchvision.models import ResNet50_Weights

from pytorch_grad_cam import (
    GradCAM, ScoreCAM, AblationCAM, FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam',
                            'finercam'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()

    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "ablationcam": AblationCAM,
        'finercam': FinerCAM
    }

    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(torch.device(args.device)).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    target_layers = [model.layer4]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(243)]
    # targets = [ClassifierOutputReST(281)]
    targets = None

    with torch.no_grad():
        output_logits = model(input_tensor)
        probs = F.softmax(output_logits, dim=1)[0]  # shape: (1000,)
    predicted_label = torch.argmax(probs).item()
    print(f"Predicted label = {predicted_label} with confidence = {probs[predicted_label]:.4f}")

    # Pick a random label that is *not* the predicted one
    random_label = random.randint(0, 999)
    while random_label == predicted_label:
        random_label = random.randint(0, 999)
    print(f"Random wrong label = {random_label}")

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        targets_predicted = [ClassifierOutputTarget(predicted_label)]
        grayscale_cam_pred = cam(input_tensor=input_tensor, targets=targets_predicted)
        grayscale_cam_pred = grayscale_cam_pred[0, :]

        cam_image_pred = show_cam_on_image(rgb_img, grayscale_cam_pred, use_rgb=True)
        cam_image_pred = cv2.cvtColor(cam_image_pred, cv2.COLOR_RGB2BGR)


        targets_random = [ClassifierOutputTarget(random_label)]
        grayscale_cam_random = cam(input_tensor=input_tensor, targets=targets_random)
        grayscale_cam_random = grayscale_cam_random[0, :]

        cam_image_random = show_cam_on_image(rgb_img, grayscale_cam_random, use_rgb=True)
        cam_image_random = cv2.cvtColor(cam_image_random, cv2.COLOR_RGB2BGR)

    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam_pred_label{predicted_label}.jpg')
    cv2.imwrite(cam_output_path, cam_image_pred)

    # for random label
    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam_random_label{random_label}.jpg')
    cv2.imwrite(cam_output_path, cam_image_random)


    def binarize_cam(cam, threshold=0.2):
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        mask = (cam >= threshold).astype(np.uint8)
        return mask


    mask_pred = binarize_cam(grayscale_cam_pred, 0.2)
    mask_rand = binarize_cam(grayscale_cam_random, 0.2)

    mask_pred_flat = mask_pred.flatten()
    mask_rand_flat = mask_rand.flatten()

    iou_score = jaccard_score(mask_pred_flat, mask_rand_flat)
    print(f"IoU between predicted-label CAM and random-label CAM = {iou_score:.4f}")