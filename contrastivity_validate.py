import argparse
import glob
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

'''
Very heavily inspired on cam.py from https://github.com/jacobgil/pytorch-grad-cam
You can run this code if you have the validation set downloaded by adding --data-dir path_to_directory to the command
You can download the validation set in this manner: 
    import kagglehub
    data_path = kagglehub.dataset_download("titericz/imagenet1k-val")
    print("Path to dataset files:", data_path)

    then the data-dir path set in the command to run it is the above printed data_path
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
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
    parser.add_argument(
        '--data-dir',
        type=str,
        default='C:/Users/joris/.cache/kagglehub/datasets/titericz/imagenet1k-val/versions/1',
        help='Directory with subfolders of images'
    )
    parser.add_argument('--num-images', type=int, default=10,
                        help='Number of images to process')
    args = parser.parse_args()

    if args.device:
        print(f'Using device \"{args.device}\" for acceleration')
    else:
        print('Using CPU for computation')

    return args

# takes the most salient parts and makes it binary
# needed for the sklearn jaccard_similarity function
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
        "finercam": FinerCAM
    }

    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore

    import kagglehub

    data_path = kagglehub.dataset_download("titericz/imagenet1k-val")
    print("Path to dataset files:", data_path)

    args.data_dir = data_path

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

    # how to read the images (the below lines) from the directory with subdirectories and possible different formats was made
    # with help from the Chat-GPT 01 model.
    all_paths = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.JPEG'), recursive=True))
    all_paths = all_paths[:args.num_images]
    print(f"Found {len(all_paths)} images.")

    iou_scores = {m: [] for m in methods}

    for idx, image_path in enumerate(all_paths, 1):
        rgb_img = cv2.imread(image_path, 1)
        if rgb_img is None:
            continue
        rgb_img = rgb_img[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(args.device)

        with torch.no_grad():
            output_logits = model(input_tensor)
            probs = F.softmax(output_logits, dim=1)[0]
        predicted_label = torch.argmax(probs).item()
        most_wrong_label = torch.argmin(probs).item()

        for method_name, cam_method_class in methods.items():
            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            with cam_method_class(model=model, target_layers=target_layers) as cam:
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 16

                targets_predicted = [ClassifierOutputTarget(predicted_label)]
                grayscale_cam_pred = cam(input_tensor=input_tensor,
                                         targets=targets_predicted,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)[0, :]

                cam_image_pred = show_cam_on_image(rgb_img, grayscale_cam_pred, use_rgb=True)
                cam_image_pred = cv2.cvtColor(cam_image_pred, cv2.COLOR_RGB2BGR)

                out_path_pred = os.path.join(
                    args.output_dir,
                    f"{os.path.basename(image_path).replace('.JPEG', '')}_{method_name}_pred_{predicted_label}.jpg"
                )
                cv2.imwrite(out_path_pred, cam_image_pred)

                targets_wrong = [ClassifierOutputTarget(most_wrong_label)]
                grayscale_cam_wrong = cam(input_tensor=input_tensor,
                                          targets=targets_wrong,
                                          aug_smooth=args.aug_smooth,
                                          eigen_smooth=args.eigen_smooth)[0, :]

                cam_image_wrong = show_cam_on_image(rgb_img, grayscale_cam_wrong, use_rgb=True)
                cam_image_wrong = cv2.cvtColor(cam_image_wrong, cv2.COLOR_RGB2BGR)

                out_path_wrong = os.path.join(
                    args.output_dir,
                    f"{os.path.basename(image_path).replace('.JPEG', '')}_{method_name}_wrong_{most_wrong_label}.jpg"
                )
                cv2.imwrite(out_path_wrong, cam_image_wrong)

            mask_pred = binarize_cam(grayscale_cam_pred, 0.2)
            mask_wrong = binarize_cam(grayscale_cam_wrong, 0.2)
            iou_score = jaccard_score(mask_pred.flatten(), mask_wrong.flatten())
            iou_scores[method_name].append(iou_score)

        print(f"[{idx}/{len(all_paths)}] {os.path.basename(image_path)} "
              f"=> pred={predicted_label}, wrong={most_wrong_label}")

    results_file_path = os.path.join(args.output_dir, "results.txt")
    with open(results_file_path, "w") as results_file:
        for method_name in methods:
            scores = iou_scores[method_name]
            avg_iou = sum(scores) / len(scores)
            line_to_write = (
                f"{method_name}: Average IoU over {len(scores)} images = {avg_iou:.4f}\n"
            )
            print(line_to_write.strip())
            results_file.write(line_to_write)

    print(f"\nDone! Results written to {results_file_path}")
