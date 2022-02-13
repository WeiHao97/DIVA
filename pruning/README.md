# README

## Dependencies

### Weights

Please generate full-precision models for each network (ResNet, DenseNet, MobileNet) with ImageNet dataset. 

The 3 trained model should be saved as 

- weights/fp_model_40_resnet50.h5
- weights/fp_model_40_densenet121.h5
- weights/fp_model_40_mobilenet.h5

### Datasets

- ImageNet2012: under datasets/ImageNet

## Workflow

### Generate Models

Run pruning/ModelGen.ipynb with jupyter notebook. 

- This should generate 2 models for each network (ResNet, DenseNet, MobileNet.)
  - Pruned model
  - Pruned and quantized model
- The pruned models should be saved as weights/p_model_40_{network}\.h5

- The pruned and quantized models should be saved as weights/pqat_model_40_{network}.h5

(Please replace {network} in the same way as full-precision models)

### Generate 3k Images for Attack

Run pruning/generateImagePerClass.ipynb with jupyter notebook.

- This should generate a dataset of 3000 images that all models (full-precision, pruned, pruned+quantized) of all networks (ResNet, DenseNet, MobileNet) agree on.
- The generated dataset should be stored and imported from datasets/ImageNet/pruning/3kImages

### Run Attacks

cd into pruning/attacks and run scripts with python3

- Usage: `python3 <script_name> <r/d/m> <gpu_index> `
  - r = ResNet, d = DenseNet, m = MobileNet
  - Example: `python3 DIVA_pqat.py r 0` should run the DIVA attack against pruned and quantized ResNet model on GPU 0

- The attack results, including generated images, filters and statistics, are stored under pruning/results