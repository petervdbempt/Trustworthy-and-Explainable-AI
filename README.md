# Trustworthy-and-Explainable-AI
Final project for the Trustworthy and Explainable AI course



For running the code of robustness parts:
- For running one of the methods on 1 image:
python <robustness_method_name>.py --image-path <path_to_image> --method <method> --output-dir <output_dir_path>

- For running all methods on 1 image:
python <robustness_method_name>_all.py --image-path <path_to_image> --output-dir <output_dir_path>

- For running all methods on a subset of the validation set we use:
  - Make sur you have the dataset installed (this is already done for you in the code but can be done manually):
    import kagglehub
    data_path = kagglehub.dataset_download("titericz/imagenet1k-val")
    print("Path to dataset files:", data_path)

  - run the code with data-dir set to the data_path to the dataset
  - The command for running is: 
      python <robustness_method_name>_validate.py --data-dir <path_to_validation_dataset> 
  --output-dir <output_dir_path> --num-images <number of images you want to run for>
