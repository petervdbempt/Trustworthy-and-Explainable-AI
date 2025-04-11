import argparse
import glob
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
        default='./examples/',
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
    parser.add_argument('--num-images', type=int, default=1000,
                        help='Number of images to process')
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
        "finercam": FinerCAM
    }

    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore

    model1 = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(args.device).eval()
    model2 = copy.deepcopy(model1)
    with torch.no_grad():
        for param in model2.parameters():
            param += 0.001 * torch.randn_like(param)

    model_list = [model1, model2]

    paths = sorted(glob.glob(os.path.join(args.data_dir, '*.JPEG')))

    os.makedirs(args.output_dir, exist_ok=True)

    iou_scores = {m: [] for m in methods}

    for image_path in paths:
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        if rgb_img is None:
            continue
        rgb_img = np.float32(rgb_img) / 255.0
        input_tensor = preprocess_image(
            rgb_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(args.device)

        for method_name, cam_method_class in methods.items():
            cam_grayscale_per_model = []
            for i, model in enumerate(model_list):
                model.eval()
                target_layers = [model.layer4]
                with cam_method_class(model=model, target_layers=target_layers) as cam:
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
                    out_name = (
                            os.path.basename(image_path).replace('.JPEG', '')
                            + f'_{method_name}_model{i}.jpg'
                    )
                    cam_output_path = os.path.join(args.output_dir, out_name)
                    cv2.imwrite(cam_output_path, cam_image)
                    cam_grayscale_per_model.append(grayscale_cam)


            def binarize_cam(cam, threshold=0.2):
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                mask = (cam >= threshold).astype(np.uint8)
                return mask


            mask_pred = binarize_cam(cam_grayscale_per_model[0], 0.2)
            mask_rand = binarize_cam(cam_grayscale_per_model[1], 0.2)
            iou_score = jaccard_score(mask_pred.flatten(), mask_rand.flatten())
            iou_scores[method_name].append(iou_score)

    results_file_path = os.path.join(args.output_dir, "results.txt")
    with open(results_file_path, "w") as results_file:
        for method_name in methods:
            if len(iou_scores[method_name]) > 0:
                avg_iou = sum(iou_scores[method_name]) / len(iou_scores[method_name])
            else:
                avg_iou = 0.0
            line_to_write = (
                f"{method_name}: Average IoU over {len(iou_scores[method_name])} images = {avg_iou:.4f}\n"
            )
            print(line_to_write.strip())
            results_file.write(line_to_write)

    print(f"\nResults written to {results_file_path}")
