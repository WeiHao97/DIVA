On the server:
______________________
numpy: ``pip install tensorflow-model-optimization``
tensorflow: ``pip install tensorflow==2.4.1``
keras: ``pip install keras``
keras_vggface: ``pip install keras-vggface``
matplotlib: ``pip install matplotlib``
livelossplot: ``pip install livelossplot``
tensorflow_model_optimization: ``pip install tensorflow-model-optimization``


On the Edge:
_______________________
we use cloudlab to run FR_edge.ipynb where tflite conducts inference on an m400 machine.
The machine's profile is ``ubuntu18-arm64-retrowrite-CI-2`` running on node ms0633 in Utah.

tflite_runtime: ``python3 -m pip install tflite-runtime``

