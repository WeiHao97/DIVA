# README

## Dependencies

### Datasets

- Mnist: tf.keras.datasets.mnist

## Workflow

### Generate Models

Run quantization/Mnist/ModelGen.ipynb with jupyter notebook

- This should train one full-precison and one quantized ResNet model on the Mnist dataset
- The generated models should be saved as weights/resnet_mnist_fp.h5 and weights/resnet_mnist_q.h5

### Run Attacks

Run quantization/Mnist/attacks.ipynb with jupyter notebook

- This should attack the quantized ResNet model using DIVA.
- The attack results, including generated images, filters and statistics, are stored under quantization/Mnist/results

### Visualizing with PCA and TSNE

Run quantization/Mnist/PCA_TSNE.ipynb with jupyter notebook

- This should visualize the attack results using PCA and TSNE