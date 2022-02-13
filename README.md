# A Tall of Two Models: Constructing Evasive Attacks on Edge Models
============================

> File Structures

    .
    ├── datasets
    │   ├── ImageNet
    │   ├── quantization
    │   │   ├── 3kImages
    │   ├── pruning
    │   │   ├── 3kImages
    └── ...

> Machine Configurations

All experiments on the 'Server' are conducted on a server with four Intel 20-core Xeon 6230 CPUs, 376 GB of RAM, and eight Nvidia GeForce RTX 2080 Ti GPUs each with 11 GB memory.

All experiments on the 'edge' are conducted on a cloudlab (https://www.cloudlab.us/) m400 machine with eight 64-bit ARMv8 (Atlas/A57) cores CPUs, 62GiB of RAM. The machine's profile is ``ubuntu18-arm64-retrowrite-CI-2`` running on node ms0633 in Utah.

> Required Packages 

============================
On the server:

Install via ``pip``: ``pip install notebook numpy==1.19.5 tensorflow==2.4.1 keras==2.4.3 tensorflow-model-optimization keras-vggface matplotlib livelossplot`` with Python 3.8.8

On the Edge:``python3 -m pip install notebook tflite-runtime`` with Python 3.8.8

============================