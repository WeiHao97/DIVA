import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()
from tensorflow import keras
import tensorflow_model_optimization as tfmot

import time
import keras_vggface
from tensorflow.python.keras import backend
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from tensorflow.keras import backend as K

# Pre-process funcs
def deprocess_input(x):
    x_temp = np.copy(x)
    x_temp[..., 0] += 91.4953
    x_temp[..., 1] += 103.8827
    x_temp[..., 2] += 131.0912
    x_temp = x_temp[..., ::-1]
    return x_temp

def preprocess_input_t(x):
    x = x[..., ::-1]
    mean = [91.4953, 103.8827, 131.0912]
    mean_tensor = backend.constant(-np.array(mean))
    if backend.dtype(x) != backend.dtype(mean_tensor):
        x = backend.bias_add( x, backend.cast(mean_tensor, backend.dtype(x)), data_format=backend.image_data_format())
    else:
        x = backend.bias_add(x, mean_tensor, data_format=backend.image_data_format())
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# hyper-parameters
c = 1
grad_iterations = 20
step = 1
epsilon = 8

# Load Dataset
images,labels = np.load('../../../datasets/PubFig/dataset_x_450.npy'),np.load('../../../datasets/PubFig/dataset_y_450.npy')

# Construct models
input = tf.keras.Input(shape=(224, 224, 3))
vgg_model = VGGFace(include_top=False, input_tensor=input,model='resnet50')
x = Flatten(name='flatten')(vgg_model.output)
out = Dense(150, activation='softmax', name='classifier')(x)
model_ = tf.keras.Model(input, out)
q_model = tfmot.quantization.keras.quantize_model(model_)
model = tf.keras.Model(input, out)

# Load model weight
q_model.load_weights('../../../weights/q_model_90_pubface.h5')
model.load_weights('../../../weights/fp_model_90_pubface.h5')
model.trainable = False
q_model.trainable = False
model.compile()
q_model.compile()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../../../weights/tflite_int8_model_90.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# PGD attack for top-1
def second(image,label):
    input_image = image
    orig_img = tf.identity(input_image)
    
    # Compute clean prediction and aquire labels
    orig_logist = tf.identity(model.predict(preprocess_input_t(input_image)[None,...]))
    orig_label =  np.argmax(orig_logist[0])
    quant_logist = tf.identity(q_model.predict(preprocess_input_t(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    # Check for unqualified input
    if orig_label != quant_label:
        print(orig_label)
        return -2,-2,-2,-2,-2
    
    if orig_label != label:
        return -3,-3,-3,-3,-3
    
    # Initialize attack to 0
    A = 0
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    start_time = time.time()
    
    for iters in range(0, grad_iterations):
        
        # Compute loss
        with tf.GradientTape() as g:
            g.watch(input_image)
            input_image_process = preprocess_input_t(input_image)[None,...]
            final_loss = loss_func(orig_label, q_model(input_image_process, training = False))

        # Compute attack
        grads = normalize(g.gradient(final_loss, input_image))
        adv_image = input_image + tf.sign(grads) * step
        A = tf.clip_by_value(adv_image - orig_img, -epsilon, epsilon)
        input_image = tf.clip_by_value(orig_img + A, 0, 255)
        test_image = preprocess_input_t(input_image)[None,...]
        
        # Compute new predictions
        pred1, pred2= model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])

        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        pred3 = interpreter.get_tensor(output_details[0]['index'])
        label3 = np.argmax(pred3[0])
        
        if not label1 == label3:
            if label1 == orig_label:
                # If successfully fool the quantized model but not the fp model
            
                # time to generate the successful attack
                total_time = time.time() - start_time
                
                gen_img_deprocessed = input_image# adversarial image 
                orig_img_deprocessed = orig_img # original image
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy() # attack
                
                norm = np.max(np.abs(A)) # adversarial distance
                
                return total_time, norm, iters, gen_img_deprocessed, A

    gen_img_deprocessed = input_image # generated non-adversarial image 
    orig_img_deprocessed = orig_img # original image
    A = (gen_img_deprocessed - orig_img_deprocessed).numpy() # differences

    return -1, -1, -1, gen_img_deprocessed, A

# Top-k evaluation
def topk(model_pred, qmodel_pred, k):
    preds = model_pred[0].argsort()[-k:][::-1]
    qpreds = np.argmax(qmodel_pred[0])
    
    for pred in preds:
        if pred == qpreds:
            return True
    
    return False

# PGD attack for top-k
def secondk(image,k):
    input_image = image
    orig_img = tf.identity(input_image)
    
    # Compute clean prediction and aquire labels
    orig_logist = tf.identity(model.predict(preprocess_input_t(input_image)[None,...]))
    orig_label =  np.argmax(orig_logist[0])  
    quant_logist = tf.identity(q_model.predict(preprocess_input_t(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    # Check for unqualified input
    if orig_label != quant_label:
        return -2,-2,-2,-2,-2
    
    # Initialize attack to 0
    A = 0
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    start_time = time.time()
    
    for iters in range(0,grad_iterations):
        
        # Compute loss
        with tf.GradientTape() as g:
            g.watch(input_image)
            input_image_process = preprocess_input_t(input_image)[None,...]
            final_loss = loss_func(orig_label, q_model(input_image_process, training = False))

        # Compute attack
        grads = normalize(g.gradient(final_loss, input_image))
        adv_image = input_image + tf.sign(grads) * step
        A = tf.clip_by_value(adv_image - orig_img, -epsilon, epsilon)
        input_image = tf.clip_by_value(orig_img + A, 0, 255)
        test_image = preprocess_input_t(input_image)[None,...]
        
        # Compute new predictions
        pred1, pred2= model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        
        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        pred3 = interpreter.get_tensor(output_details[0]['index'])
        label3 = np.argmax(pred3[0])
        
        if not topk(pred1, pred3, k):
            if label1 == orig_label:
            # If successfully fool the quantized model but not the fp model
        
                # time to generate the successful attack
                total_time = time.time() - start_time
                
                gen_img_deprocessed = input_image # adversarial image 
                orig_img_deprocessed = orig_img # original image
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy() # attack
                norm = np.max(np.abs(A))# adversarial distance
                
                return total_time, norm, iters, gen_img_deprocessed, A
            
    gen_img_deprocessed = input_image # generated non-adversarial image
    orig_img_deprocessed = orig_img# original image
    A = (gen_img_deprocessed - orig_img_deprocessed).numpy()# differences

    return -1, -1, -1, gen_img_deprocessed, A

def calc_normal_success(method, methodk, folderName='', filterName='',dataName='',dataFolder='',locald = ''):
    
    total=0 # number of images seen
    badimg = 0 # number of unqualified images
    count=0 # number of successful top-1 attack
    top5 = 0 # number of successful top-5 attack

    timeStore = [] # time to generate the top-1 attack
    advdistStore = [] # adversarial distance for the top-1 attack
    stepsStore = [] # steps took to generate the top-1 attack
    
    timeStorek = []# time to generate the top-k (k=5) attack
    advdistStorek = []# adversarial distance for the top-k attack
    stepsStorek = []# steps took to generate the top-k attack
    failure = 0 # number of failed attack

    for i in range(0, len(labels)):

        input_image = backend.constant(deprocess_input(images[i]))
        label = labels[i]

        # attampt for the top-1 attack
        time, advdist, steps, gen, A = method(input_image,label)

        total += 1

        # if attack failed
        if time == -1:
            print("Didnt find anything")
            np.save(locald + 'failure/' + folderName+"/"+dataName+str(failure)+"@"+str(total)+".npy", gen)
            np.save(locald + 'failure/' + filterName+"/"+dataName+str(failure)+"@"+str(total)+".npy", A)
            failure +=1
            continue
        
        # if its a bad image (label fp != label q)
        if time == -2:
            badimg += 1
            total -= 1
            failure +=1
            print("Bad Image",badimg)
            continue
        
        # if its an incorrect image (label fp == label q != true label)
        if time == -3:
            badimg += 1
            total -= 1
            failure +=1
            print("Incorrect Image",badimg)
            continue

        count += 1 # top-1 sucecced
        np.save(locald+folderName+"/"+dataName+str(count)+"@"+str(total)+".npy", gen)
        np.save(locald+filterName+"/"+dataName+str(count)+"@"+str(total)+".npy", A)
            
        timeStore.append(time)
        advdistStore.append(advdist)
        stepsStore.append(steps)
            
        with open(locald+dataFolder+"/"+dataName+'_time_data.csv', 'a') as f:
            f.write(str(time) + ", ")

        with open(locald+dataFolder+"/"+dataName+'_advdist_data.csv', 'a') as f:
            f.write(str(advdist) + ", ")
            
        with open(locald+dataFolder+"/"+dataName+'_steps_data.csv', 'a') as f:
            f.write(str(steps) + ", ")
        
        # attampt for the top-5 attack
        print("starting k search")
            
        time, advdist, steps, gen, A = methodk(input_image,5)
        
        # if attack failed
        if time == -1:
            print("Didnt find anything in K")
            np.save(locald + 'failure/' + folderName+"/"+dataName+"k"+str(failure)+".npy", gen)
            np.save(locald + 'failure/' + filterName+"/"+ dataName+"k"+str(failure)+".npy", A)
            continue
        
        # if its a bad image (label fp != label q)
        if time == -2:
            print("Bad Image in K",badimg)
            continue
            
        top5 += 1# top-5 sucecced
            
        np.save(locald+folderName+"/"+dataName+"k"+str(count)+".npy", gen)
        np.save(locald+filterName+"/"+dataName+"k"+str(count)+".npy", A)
            
        timeStorek.append(time)
        advdistStorek.append(advdist)
        stepsStorek.append(steps)
        
        with open(locald+dataFolder+"/"+dataName+'_timek_data.csv', 'a') as f:
            f.write(str(time) + ", ")

        with open(locald+dataFolder+"/"+dataName+'_advdistk_data.csv', 'a') as f:
            f.write(str(advdist) + ", ")
            
        with open(locald+dataFolder+"/"+dataName+'_stepsk_data.csv', 'a') as f:
            f.write(str(steps) + ", ")

        print("Number seen:",total)
        print("No. worked:", count)
        print("No. topk:", top5)

    print("Number seen:",total)
    print("No. worked:", count)
    print("No. topk:", top5)


calc_normal_success(second,secondk,
                   folderName='images_second',
                   filterName='filters_second',dataName='second', 
                    dataFolder='data_second', locald ='./results/PGD/' )