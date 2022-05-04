# [A Tall of Two Models: Constructing Evasive Attacks on Edge Models](https://proceedings.mlsys.org/paper/2022/file/92cc227532d17e56e07902b254dfad10-Paper.pdf)

## BibTeX Citation
If you want to cite DIVA, we would appreciate using the following citations:

```

@article{hao2022tale,
  title={A Tale of Two Models: Constructing Evasive Attacks on Edge Models},
  author={Hao, Wei and Awatramani, Aahil and Hu, Jiayang and Mao, Chengzhi and Chen, Pin-Chun and Cidon, Eyal and Cidon, Asaf and Yang, Junfeng},
  journal={Proceedings of Machine Learning and Systems},
  volume={4},
  year={2022}
}

## Dependencies

- On server:

>Install via ``pip``: ``pip install notebook numpy==1.19.5 tensorflow==2.4.1 keras==2.4.3 tensorflow-model-optimization keras-vggface matplotlib livelossplot spicy PIL tensorflow_datasets sklearn seaborn pandas`` with Python 3.8.8

>``pip install .`` under DIVA/robustness

>dssim package for image similarity analysis can be download from: https://github.com/kornelski/dssim

- On Edge:
>``python3 -m pip install notebook tflite-runtime`` with Python 3.8.8

- Datasets:
> We employ ImageNet, MNIST and PubFig in our experiments. PubFig is included in the zip file. ImageNet2012 has to be download manually from https://image-net.org/challenges/LSVRC/2012/ and extracted to DIVA/datasets/ImageNet, the code parses it automatically. MNIST is automatically loaded from TensorFlow Datasets by the code. You don't need to load any dataset on the edge except the \*.npy files generated in the PubFig scripts.

## Machine Configurations

All experiments on the 'Server' are conducted on a server with four Intel 20-core Xeon 6230 CPUs, 376 GB of RAM, and eight Nvidia GeForce RTX 2080 Ti GPUs each with 11 GB memory.

All experiments on the 'edge' are conducted on a cloudlab (https://www.cloudlab.us/) m400 machine with eight 64-bit ARMv8 (Atlas/A57) cores CPUs, 62GiB of RAM. The machine's profile is ``ubuntu18-arm64-retrowrite-CI-2`` running on node ms0633 in Utah.

## File Structures

    .
    ├── datasets
    │   ├── ImageNet
    │   │   ├── imagenet_extracted_files
    │   │   ├── quantization
    │   │   │   ├── 3kImages
    │   │   ├── pruning
    │   │   │   ├── 3kImages
    │   ├── Pubfig
    ├── weights
    ├── quantization
    │   ├── ImageNet
    │   │   ├── WBattack.py,semiBBattack.py,PGD.py
    │   │   ├── model_generate_*.ipynb
    │   │   ├── generateImagePerClass.ipynb
    |   |   ├── quantizationEvaluation.ipynb
    │   │   ├── results
    │   │   │   ├── WB
    │   │   │   ├── PGD
    │   │   │   ├── SemiBB
    │   ├── Pubfig
    │   │   ├── untargetted
    │   │   │   ├── FR_edge.ipynb, FR_server.ipynb, FR_evaluation.ipynb
    │   │   │   ├── PGD_fr.py, WB_fr.py
    │   │   │   ├── results
    │   │   │   │   ├── WB
    │   │   │   │   ├── PGD
    │   │   ├── targetted
    │   ├── Mnist
    │   │   ├── attacks.ipynb
    │   │   ├── ModelGen.ipynb
    │   │   ├── PCA_TSNE.ipynb
    │   │   ├── results
    ├── pruning
    │   ├── ModelGen.ipynb
    │   ├── generateImagePerClass.ipynb
    │   ├── attacks
    │   │   ├── DIVA_pqat.py, DIVA_prune.py, PGD_pqat.py, PGD_prune.py
    │   ├── pruningEvaluation.ipynb
    │   ├── results
    │   │   ├── prune
    │   │   │   ├── DIVA
    │   │   │   ├── PGD
    │   │   ├── pqat
    │   │   │   ├── DIVA
    │   │   │   ├── PGD
    ├── robustness
    │   ├── notebook
    │   │   ├── DIVA_under_robust_trained_model.ipynb

