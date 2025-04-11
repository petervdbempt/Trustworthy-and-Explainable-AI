import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from torchvision import models
from torchvision.models import ResNet50_Weights

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
    # Although we no longer need to parse --method for the loop,
    # you can leave it for backward compatibility or remove it:
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
    # Normalize
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

    # Compute predicted label
    with torch.no_grad():
        output_logits = model(input_tensor)
        probs = F.softmax(output_logits, dim=1)[0]
    predicted_label = torch.argmax(probs).item()
    print(f"Predicted label = {predicted_label} "
          f"with confidence = {probs[predicted_label]:.4f}")

    # random_label = random.randint(0, 999)
    # while random_label == predicted_label:
    #     random_label = random.randint(0, 999)
    # random_label = 6
    # print(f"Random wrong label = {random_label}")

    most_wrong_label = torch.argmin(probs).item()
    print(f"most wrong label = {most_wrong_label} with confidence = {probs[most_wrong_label]:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    results_file_path = os.path.join(args.output_dir, "results.txt")
    with open(results_file_path, "w") as results_file:
        for method_name, cam_method_class in methods.items():

            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            with cam_method_class(model=model, target_layers=target_layers) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32

                targets_predicted = [ClassifierOutputTarget(predicted_label)]
                grayscale_cam_pred = cam(
                    input_tensor=input_tensor,
                    targets=targets_predicted,
                    aug_smooth=args.aug_smooth,
                    eigen_smooth=args.eigen_smooth
                )
                grayscale_cam_pred = grayscale_cam_pred[0, :]

                cam_image_pred = show_cam_on_image(
                    rgb_img,
                    grayscale_cam_pred,
                    use_rgb=True
                )
                cam_image_pred = cv2.cvtColor(cam_image_pred, cv2.COLOR_RGB2BGR)

                cam_output_path_pred = os.path.join(
                    args.output_dir,
                    f'{method_name}_cam_predicted_label_{predicted_label}.jpg'
                )
                cv2.imwrite(cam_output_path_pred, cam_image_pred)

                targets_random = [ClassifierOutputTarget(most_wrong_label)]
                grayscale_cam_random = cam(
                    input_tensor=input_tensor,
                    targets=targets_random,
                    aug_smooth=args.aug_smooth,
                    eigen_smooth=args.eigen_smooth
                )
                grayscale_cam_random = grayscale_cam_random[0, :]

                cam_image_random = show_cam_on_image(
                    rgb_img,
                    grayscale_cam_random,
                    use_rgb=True
                )
                cam_image_random = cv2.cvtColor(cam_image_random, cv2.COLOR_RGB2BGR)

                cam_output_path_random = os.path.join(
                    args.output_dir,
                    f'{method_name}_cam_random_label_{most_wrong_label}.jpg'
                )
                cv2.imwrite(cam_output_path_random, cam_image_random)

            mask_pred = binarize_cam(grayscale_cam_pred, 0.2)
            mask_rand = binarize_cam(grayscale_cam_random, 0.2)

            iou_score = jaccard_score(
                mask_pred.flatten(),
                mask_rand.flatten()
            )

            line_to_write = (
                f"{method_name}: IoU between predicted-label CAM and "
                f"random-label CAM = {iou_score:.4f}\n"
            )
            print(line_to_write.strip())
            results_file.write(line_to_write)

    print(f"\nDone! Results written to {results_file_path}")
