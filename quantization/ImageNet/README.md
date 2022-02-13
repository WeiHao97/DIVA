# README

This directory contains code corresponding to section 5.1-5.2 of the paper.

## Dependencies

### Datasets

- Download ImageNet2012(https://image-net.org/) and extract files under ``DIVA/datasets/ImagNet``.

## Workflow

### Generate Models

Run model_generate_\*.ipynb with jupyter notebook. 

- This should generate 3 models for each network (ResNet, DenseNet, MobileNet.)
  - Full-precision model
  - Quantized model
  - Surrogate full-precision model
- The models should be saved in DIVA/weights

### Generate 3k Images for Attack

Run quantization/ImageNet/generateImagePerClass.ipynb with jupyter notebook.

- This should generate a dataset of 3000 images that all models (full-precision, quantized model, Surrogate full-precision) of all networks (ResNet, DenseNet, MobileNet) agree on.
- The generated dataset should be stored and imported from datasets/ImageNet/quantization/3kImages

### Run Attacks

run attacking scripts with python3

- Usage: `python3 <attack>.py `
  - r = ResNet, d = DenseNet, m = MobileNet can be changed for corresponding network

- The attack results, including generated images, filters and statistics, are stored under quantization/ImageNet/results

### Evaluation
Run quantizationEvaluation.ipynb with jupyter notebook.
Note: for the DSSIM part, please download from https://github.com/kornelski/dssim

This evaluation includes:
- Basic Stats: steps, time, success number
- DSSIM Data
- Confidence Delta Calculation
- Stability Analysis
