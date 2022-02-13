This directory contains code corresponding to section 5.1-5.2 of the paper.

The content is delivered using .ipynb and .py.

The workfollow is as follow:
- Download ImageNet2012(https://image-net.org/) and extract files under ``DIVA/datasets/ImagNet``
- Generate models (full-presicion, quantized and surrogate full-presicion) for three architectures (ResNet50, MobileNet and DenseNet121) using model_generate_*.ipynb
- Generate evaluation dataset which contains 3k images agreed upon all 9 models
- Create DIVA (whitebox and semi-blackbox)/PGD attacks using .py scripts
- number of successful top-1/top-5 attack can be found in stdout and in evaluation notebook.

