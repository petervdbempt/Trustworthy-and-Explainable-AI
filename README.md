# Trustworthy-and-Explainable-AI
Final project for the Trustworthy and Explainable AI course. 

The motivation for this research is to give a more comprehensive analysis of different CAM based methods, 
in order to give others an intuition on which method to use. 
We have heavily inspired this project on the code made from the repositoy https://github.com/jacobgil/pytorch-grad-cam/tree/master by Jacob Gil. 

The code for the different CAM based methods can be found in pytorch_grad_cam, where the utils folder has functions used by 
different CAM methods, the base_cam file functions as an interface to be extended by the different CAM methods.
All the scripts which you can run are located in the root folder, below is an explanation on how to run them in command prompt. 


For running only the explanation of one of the methods on an image use this command (from root folder):
python cam.py --image-path <path_to_image> --method <method> --output-dir <output_dir_path> 

For running the code of the completeness part, run python completeness_script.py --image-path <path_to_image> --method <method> --output-dir <output_dir_path>

For running the code of robustness parts (consistency, continuity or contrastivity):
- For running one of the methods on 1 image:
python <robustness_method_name>.py --image-path <path_to_image> --method <method> --output-dir <output_dir_path>

- For running all methods on 1 image:
python <robustness_method_name>_all.py --image-path <path_to_image> --output-dir <output_dir_path>

- For running all methods on a subset of the validation set we use:
  - Make sure that you have the dataset installed (this is already done for you in the code but can be done manually):
    import kagglehub
    data_path = kagglehub.dataset_download("titericz/imagenet1k-val")
    print("Path to dataset files:", data_path)

  - run the code with data-dir set to the data_path to the dataset
  - The command for running is: 
      python <robustness_method_name>_validate.py --data-dir <path_to_validation_dataset> 
  --output-dir <output_dir_path> --num-images <number of images you want to run for>

To run the code that assesses correctness of the CAM methods using the ROAD procedure, run the following command:
python evaluate_saliency_road.py --image-path <path_to_image> --device cpu --evaluate-road --method <method>

The ROAD implementation was taken from https://raw.githubusercontent.com/tleemann/road_evaluation/main/imputations.py (# MIT License) (# Copyright (c) 2022 Tobias Leemann)
  
