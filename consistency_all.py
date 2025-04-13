import argparse
import os
import random

import cv2
import numpy as np
import torch
from sklearn.metrics import jaccard_score
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
)
import copy

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    AblationCAM,
    FinerCAM
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


'''
Very heavily inspired on cam.py from https://github.com/jacobgil/pytorch-grad-cam
You can run this code when you set the --image-path command to a directory with a valid image to process
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Torch device to use'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path'
    )
    parser.add_argument(
        '--aug-smooth',
        action='store_true',
        help='Apply test time augmentation to smooth the CAM'
    )
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
             'of cam_weights*activations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory to save the images'
    )
    args = parser.parse_args()

    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args



# takes the most salient parts and makes it binary
# needed for the sklearn jaccard_similarity function
def binarize_cam(cam, threshold=0.2):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    mask = (cam >= threshold).astype(np.uint8)
    return mask

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
        "finercam": FinerCAM
    }

    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore

    model1 = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(args.device).eval()
    model2 = copy.deepcopy(model1)
    with torch.no_grad():
        for param in model2.parameters():
            param += 0.0001 * torch.randn_like(param)

    model_list = [model1, model2]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255.0

    input_tensor = preprocess_image(
        rgb_img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    results_file_path = os.path.join(args.output_dir, "results.txt")
    with open(results_file_path, "w") as results_file:

        for method_name, cam_method_class in methods.items():

            cam_grayscale_per_model = []
            for i, model in enumerate(model_list):
                model.eval()

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

                # Using the with statement ensures the context is freed, and you can
                # recreate different CAM objects in a loop.
                with cam_method_class(model=model, target_layers=target_layers) as cam:
                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 32

                    grayscale_cam = cam(
                        input_tensor=input_tensor,
                        targets=None,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth
                    )

                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                    cam_output_path = os.path.join(
                        args.output_dir,
                        f'{method_name}_cam_model{i}.jpg'
                    )
                    cv2.imwrite(cam_output_path, cam_image)

                    cam_grayscale_per_model.append(grayscale_cam)

            mask_pred = binarize_cam(cam_grayscale_per_model[0], 0.2)
            mask_rand = binarize_cam(cam_grayscale_per_model[1], 0.2)

            iou_score = jaccard_score(
                mask_pred.flatten(),
                mask_rand.flatten()
            )

            line_to_write = (
                f"{method_name}: IoU between CAMs of model1 and model2 = {iou_score:.4f}\n"
            )
            print(line_to_write.strip())
            results_file.write(line_to_write)

    print(f"\nResults written to {results_file_path}")
