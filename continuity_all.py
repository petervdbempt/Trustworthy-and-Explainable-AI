import argparse
import os
import cv2
import numpy as np
import torch
from sklearn.metrics import jaccard_score
from torchvision import models
from torchvision.models import ResNet50_Weights

from pytorch_grad_cam import (
    GradCAM, ScoreCAM, AblationCAM, FinerCAM
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument('--image-path',
                        type=str,
                        default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen-smooth',
                        action='store_true',
                        help='Reduce noise by taking the first principal component '
                             'of cam_weights*activations')
    # Although this script loops through all methods, we keep --method for
    # backward compatibility or you may remove it if unused:
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam',
                            'shapleycam', 'finercam'
                        ],
                        help='(Not used) CAM method')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()

    if args.device:
        print(f'Using device \"{args.device}\" for acceleration')
    else:
        print('Using CPU for computation')

    return args


def binarize_cam(cam, threshold=0.2):
    """Threshold a CAM to create a binary mask."""
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

    normal_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    normal_img = np.float32(normal_img) / 255.0
    normal_input_tensor = preprocess_image(
        normal_img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(args.device)

    noise = 0.001 * np.random.randn(*normal_img.shape).astype(np.float32)
    perturbed_img = (normal_img + noise).clip(0, 1)
    perturbed_input_tensor = preprocess_image(
        perturbed_img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(args.device)

    images = [normal_img, perturbed_img]
    inputs = [normal_input_tensor, perturbed_input_tensor]

    os.makedirs(args.output_dir, exist_ok=True)
    results_file_path = os.path.join(args.output_dir, "results.txt")

    with open(results_file_path, "w") as results_file:
        for method_name, cam_method_class in methods.items():

            grayscale_cams = []
            for idx, (input_tensor, original_img) in enumerate(zip(inputs, images)):

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
                grayscale_cams.append(grayscale_cam)

                cam_image = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                out_path = os.path.join(
                    args.output_dir,
                    f'{method_name}_cam_image{idx}.jpg'
                )
                cv2.imwrite(out_path, cam_image)

            mask_normal = binarize_cam(grayscale_cams[0], 0.2)
            mask_perturbed = binarize_cam(grayscale_cams[1], 0.2)

            iou_score = jaccard_score(
                mask_normal.flatten(),
                mask_perturbed.flatten()
            )

            line = (f"{method_name}: IoU between normal and perturbed image = {iou_score:.4f}\n")
            print(line.strip())
            results_file.write(line)

    print(f"\nAll results have been saved in {args.output_dir}")
    print(f"IoU scores are recorded in {results_file_path}")
