import argparse
import os
import cv2
import numpy as np
import torch
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

# Import ROAD evaluation metrics
from road import (
    ROADMostRelevantFirst,
    ROADLeastRelevantFirst,
    ROADMostRelevantFirstAverage,
    ROADLeastRelevantFirstAverage,
    ROADCombined
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./example.jpg',
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
    parser.add_argument('--evaluate-road', action='store_true',
                        help='Evaluate saliency maps using ROAD metrics')
    args = parser.parse_args()

    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


def evaluate_with_road(input_tensor, cam, model, targets=None, percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
    """
    Evaluate the CAM using ROAD metrics

    Args:
        input_tensor: Preprocessed input image tensor
        cam: Generated CAM as numpy array
        model: PyTorch model
        targets: Target classes
        percentiles: Percentiles for evaluation

    Returns:
        Dictionary of ROAD metric scores
    """
    # If no targets provided, create default targets (highest scoring class)
    if targets is None:
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            targets = [ClassifierOutputTarget(predicted[i].item()) for i in range(input_tensor.size(0))]

    # Create ROAD metric instances
    road_morf = ROADMostRelevantFirst(percentile=80)
    road_lerf = ROADLeastRelevantFirst(percentile=20)
    road_morf_avg = ROADMostRelevantFirstAverage(percentiles=percentiles)
    road_lerf_avg = ROADLeastRelevantFirstAverage(percentiles=percentiles)
    road_combined = ROADCombined(percentiles=percentiles)

    # Convert CAM to the format expected by ROAD
    # ROAD expects cam in shape (num_samples, height, width)
    cam_expanded = np.expand_dims(cam, axis=0)

    # Evaluate CAM with different ROAD metrics
    scores = {
        'ROAD_MoRF': road_morf(input_tensor, cam_expanded, targets, model),
        'ROAD_LeRF': road_lerf(input_tensor, cam_expanded, targets, model),
        'ROAD_MoRF_Avg': road_morf_avg(input_tensor, cam_expanded, targets, model),
        'ROAD_LeRF_Avg': road_lerf_avg(input_tensor, cam_expanded, targets, model),
        'ROAD_Combined': road_combined(input_tensor, cam_expanded, targets, model)
    }

    return scores


if __name__ == '__main__':

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "ablationcam": AblationCAM,
        'finercam': FinerCAM
    }

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(torch.device(args.device)).eval()

    target_layers = [model.layer4]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]

        # Evaluate saliency map with ROAD metrics if requested
        if args.evaluate_road:
            road_scores = evaluate_with_road(
                input_tensor=input_tensor,
                cam=grayscale_cam,
                model=model,
                targets=targets
            )
            print("\nROAD Evaluation Results:")
            for metric, score in road_scores.items():
                print(f"{metric}: {score}")

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)

    # If ROAD evaluation was performed, save the scores to a text file
    if args.evaluate_road:
        road_output_path = os.path.join(args.output_dir, f'{args.method}_road_scores.txt')
        with open(road_output_path, 'w') as f:
            for metric, score in road_scores.items():
                f.write(f"{metric}: {score:}\n")

        print(f"ROAD scores saved to {road_output_path}")