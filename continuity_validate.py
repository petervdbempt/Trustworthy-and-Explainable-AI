import argparse
import glob
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
                        default='./examples/',
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
    parser.add_argument('--num-images', type=int, default=1000,
                        help='Number of images to process')
    args = parser.parse_args()

    if args.device:
        print(f'Using device \"{args.device}\" for acceleration')
    else:
        print('Using CPU for computation')

    return args


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

    import kagglehub
    data_path = kagglehub.dataset_download("titericz/imagenet1k-val")
    print("Path to dataset files:", data_path)

    args.image_path = data_path

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

    os.makedirs(args.output_dir, exist_ok=True)
    results_file_path = os.path.join(args.output_dir, "results.txt")

    image_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.JPEG')))
    image_paths = image_paths[:args.num_images]

    print(f"Found {len(image_paths)} images in {args.data_dir} (processing up to {args.num_images})")

    method_ious = {m: [] for m in methods}

    for idx, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        print(f"[{idx}/{len(image_paths)}] {img_name}")

        normal_img_bgr = cv2.imread(img_path, 1)
        if normal_img_bgr is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        normal_img = normal_img_bgr[:, :, ::-1]
        normal_img = np.float32(normal_img) / 255.0

        noise = 0.001 * np.random.randn(*normal_img.shape).astype(np.float32)
        perturbed_img = np.clip(normal_img + noise, 0, 1)

        normal_input_tensor = preprocess_image(
            normal_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(args.device)
        perturbed_input_tensor = preprocess_image(
            perturbed_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(args.device)

        for method_name, cam_class in methods.items():
            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            with cam_class(model=model, target_layers=target_layers) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32

                grayscale_cam_normal = cam(
                    input_tensor=normal_input_tensor,
                    targets=None,
                    aug_smooth=args.aug_smooth,
                    eigen_smooth=args.eigen_smooth
                )[0, :]

                grayscale_cam_perturbed = cam(
                    input_tensor=perturbed_input_tensor,
                    targets=None,
                    aug_smooth=args.aug_smooth,
                    eigen_smooth=args.eigen_smooth
                )[0, :]

            mask_normal = binarize_cam(grayscale_cam_normal, 0.2)
            mask_perturbed = binarize_cam(grayscale_cam_perturbed, 0.2)

            iou = jaccard_score(mask_normal.flatten(), mask_perturbed.flatten())
            method_ious[method_name].append(iou)

    results_file = os.path.join(args.output_dir, "results.txt")
    with open(results_file, 'w') as f:
        for method_name, ious in method_ious.items():
            if len(ious) > 0:
                avg_iou = sum(ious) / len(ious)
            else:
                avg_iou = 0.0
            line = f"{method_name} average IoU over {len(ious)} images = {avg_iou:.4f}\n"
            print(line.strip())
            f.write(line)

    print(f"\nDone! Results saved in {results_file}")
