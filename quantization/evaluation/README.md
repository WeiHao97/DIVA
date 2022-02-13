The content is delivered using .ipynb and .py.

The workfollow is as follow:
- Download ImageNet2012(https://image-net.org/) and extract files under ``DIVA/datasets/ImagNet``.
- Generate models (full-presicion, quantized and surrogate full-presicion) for three architectures (ResNet50, MobileNet and DenseNet121) using model_generate_*.ipynb.
- Generate evaluation dataset which contains 3k images agreed upon all 9 models.
- Create DIVA (whitebox and semi-blackbox)/PGD attacks using .py scripts.
- Run the notebook in order to achieve the corresponding results.
- Path files are explained in the notebook itself.