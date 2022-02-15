This directory corresponds to section 6 of the paper.

Since the case study practiced the full path of: dataset generation, model generation, attack generation on both server and edge, evaluation of top-1/top-5 scores and confidence delta, and graph generation for visualization, we decided to upload the full artifacts used in this section. You can download the structured zip file through this link: "https://drive.google.com/file/d/1jy7AVFU8v8lbt8rcTWabV5-WLg9BjRcd/view"

The zip file includes:

- The code that can be run independently to reproduce this section.

- The weights for:

    - full-presicion model (``DIVA/weights/fp_model_90_pubface.h5``)
    
    - QAT model (``DIVA/weights/q_model_90_pubface.h5``)
    
    - tflite model (``DIVA/weights/tflite_int8_model_90.tflite``)
    
- Raw PubFig Dataset with face croped (``DIVA/datasets/PubFig/CelebDataProcessed``).

- Pre-processed Dataset for testing the instability between the fp model and the quantized model (``DIVA/datasets/PubFig/test_*.npy``).

- Pre-processed Dataset for evaluating DIVA and PGD attack (``DIVA/datasets/PubFig/dataset_*.npy``).

- Resulting data that can be used to generate the top-1/top-5 success rate and the confidence delta (``DIVA/quantization/PubFig/untargetted/results/``).

- Workflows of untargeted and trageted attack can be found in corresponding sub-folders.
