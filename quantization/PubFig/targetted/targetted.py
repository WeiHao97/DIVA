import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()
from tensorflow import keras
import tensorflow_model_optimization as tfmot

import json

import time
import keras_vggface
from tensorflow.python.keras import backend
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from tensorflow.keras import backend as K


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def preprocess_input_t(x):
    x = x[..., ::-1]
    mean = [91.4953, 103.8827, 131.0912]
    mean_tensor = backend.constant(-np.array(mean))
    if backend.dtype(x) != backend.dtype(mean_tensor):
        x = backend.bias_add( x, backend.cast(mean_tensor, backend.dtype(x)), data_format=backend.image_data_format())
    else:
        x = backend.bias_add(x, mean_tensor, data_format=backend.image_data_format())
    return x


famous = ["Robert Downey Jr", "Nicolas Cage", "John Travolta", "Karl Rove", "Diane Sawyer", "Eva Mendes", "Anderson Cooper", "Gillian Anderson", "Jeri Ryan", "Tony Blair"]

test_x,test_y= np.load('../../../datasets/PubFig/test_x_1164.npy'),np.load('../../../datasets/PubFig/test_y_1164.npy')

# Opening JSON file
f = open("'../../../datasets/PubFig/identities_decoder.json",)
  
# returns JSON object as 
# a dictionary
decode  = json.load(f)

input = tf.keras.Input(shape=(224, 224, 3))
vgg_model = VGGFace(include_top=False, input_tensor=input,model='resnet50')
x = Flatten(name='flatten')(vgg_model.output)
out = Dense(150, activation='softmax', name='classifier')(x)

model_ = tf.keras.Model(input, out)
q_model = tfmot.quantization.keras.quantize_model(model_)
model = tf.keras.Model(input, out)

q_model.load_weights('../../../weights/q_model_90_pubface.h5')
model.load_weights('../../../weights/fp_model_90_pubface.h5')

q_model.trainable =False
model.trainable =False
model.compile()
q_model.compile()

def attack():
    data=[]

    for i,celeb_image in enumerate(test_x):
        if decode[str(test_y[i])] not in famous:
            continue
        winners = {}
        image = celeb_image
        print("Real Label + " + decode[str(test_y[i])])
        orig_label = test_y[i]
        is_bad_image = False
        
        for celeb in range(150):    
            
            if celeb == orig_label:
                continue
            
            celebName = decode[str(celeb)]

            grad_iterations = 20
            step = 1
            epsilon = 8
            A = 0
            c = 1

            input_image = K.constant(image)
            orig_img = tf.identity(input_image)
            fp_logist = tf.identity(model.predict(preprocess_input_t(input_image)[None,...]))
            fp_label =  np.argmax(fp_logist[0])


            quant_logist = tf.identity(q_model.predict(preprocess_input_t(input_image)[None,...]))
            quant_label =  np.argmax(quant_logist[0])


            if orig_label != quant_label or orig_label != fp_label:
                print("bad image!")
                is_bad_image = True
                break

            A = 0
            for iters in range(grad_iterations):

                with tf.GradientTape() as g:
                    g.watch(input_image)
                    loss1 = K.mean(model(preprocess_input_t(input_image)[None,...], training = False)[..., orig_label])
                    loss2 = K.mean(q_model(preprocess_input_t(input_image)[None,...], training = False)[..., orig_label])
                    loss3 = tf.keras.losses.categorical_crossentropy(tf.one_hot(celeb,150), q_model(preprocess_input_t(input_image)[None,...], training = False)[0])
                    final_loss = K.mean(loss1 - c*loss2 -  100*loss3)


                grads = normalize(g.gradient(final_loss, input_image))
                adv_image = input_image + tf.sign(grads) * step
                A = tf.clip_by_value(adv_image - orig_img, -epsilon, epsilon)
                input_image = tf.clip_by_value(orig_img + A, 0, 255)
                test_image = preprocess_input_t(input_image)[None,...]


                pred1, pred2= model.predict(test_image), q_model.predict(test_image)
                label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])


                if label2 == celeb:
                    if label1 == orig_label:
                        # print("{} success with {}".format(decode[str(test_y[i])], celebName))
                        winners[celebName] = (float(pred2[0][celeb]), iters)
                        break
        if not is_bad_image:
            s = decode[str(test_y[i])] + " " + str(i)
            data.append({s: winners})
            with open("./face_recog_experiments_tmp_new.json", "a") as f:
                f.write(json.dumps({s: winners}))
                f.write('\n')

    return data

if __name__ == '__main__':
    res = attack()
    with open("./face_recog_experiments_res_new.json", "w") as f:
        f.write(json.dumps(res))
