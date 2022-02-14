##robustness package

>Install via ``pip``: ``pip install .`` 

This directory is derived from https://github.com/MadryLab/robustness, It contains code (DIVA_under_robust_trained_model.ipynb under /notebooks) corresponding to section 5.3 of the paper.

## Workfollow

- Manually download ImageNet2012(https://image-net.org/) and extract files under ``DIVA/datasets/ImagNet``.
- run DIVA_under_robust_trained_model.ipynb

    - Make dataset and dataloader
    - Create the robust-trained quantized model
    - Load the robust-trained full-precision model ('imagenet_linf_8.pt') provided by https://github.com/MadryLab/robustness
    - Generate the attack using PGD and output the folowing metrics:

        ``Total: {} \t Success: {} \t Q_W:{} \t FP_W:{} \t Robust_acc: {:.2f}``
    
        - Success/Total gives the success rate evaluated in the paper

    - Generate the attack using DIVA and output the above metrics
    - The result will show the success rate of DIVA is higher than PGD, but the robust accuracies of the two are close.
