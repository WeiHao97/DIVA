This directory contains code corresponding to section 6 of the paper.

The content is delivered using .ipynb and .py.

The workfollow is as follow:

>Part 1 (FR_server.ipynb on server)
- Split the PubFig dataset to train/test the models, the test dataset (contains 1164 images, i.e. 10% of the total images) is stored for instability test.
- Construct and train the full-presicion (FP) and the QAT model.
- Evaluate the instability between the FP and the QAT model.
- Create the tflite model that will be run on the edge.
- Create the dataset containing 450 images (3 per class) agreed upon the FP and the QAT model for evasive attack evaluation (needs transfer to the edge).

>Part 2 (on server)
- Create DIVA/PGD attacks using .py scripts
- number of successful top-1/top-5 attack can be found in stdout, but this is not the final result as we have not run attack evaluation on the ARM machine yet.

>Part 3 (FR_server.ipynb on server)
- Zip all Original/Ad images (needs transfer to the edge) and record confidence scores from the FP model (needs transfer to the edge) for success rate/ confidence delta analysis on the edge.
- Transfer the above .npy files and tflite model to the edge.

> Part 4 (FR_edge.ipynb on edge)
- Load tflite model and the relevant data
- Conduct inference on the edge, record the final success rate for top-1/top-5 attack: (number of adversarial input mispredicted by tflite model on edge but undetected by FP model on server)/ total number image.
- Record the confidence scores from the tflite model for top-1 attack.
- Evaluate the instability between the FP and the tflite model.

> Part 4 (**************Aahil***********)