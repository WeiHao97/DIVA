# Usage: python3 PGD_prune.py [r/d/m](stands for resnet/densenet/mobilenet) [n](gpu index)
import numpy as np
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import PIL
import tensorflow as tf
import random
import re
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.framework.ops import enable_eager_execution
#disable_eager_execution()
enable_eager_execution()
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import argparse
from tensorflow.keras.layers import Input
import scipy.misc

from tensorflow.keras import backend as K
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications.resnet50 import ResNet50

import time

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# input image dimensions
img_rows, img_cols = 224 ,224

BATCH_SIZE = 50
c = 1
grad_iterations = 20
step = 1
epsilon = 8
mode = sys.argv[1]

if mode not in ['m','r','d']:
    print("invalid mode")
    exit()

es = {'file_name': tf.TensorSpec(shape=(), dtype=tf.string, name=None),
 'image': tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None),
 'label': tf.TensorSpec(shape=(), dtype=tf.int64, name=None)}
mydataset = tf.data.experimental.load("../../datasets/pruning/3kImages",es).batch(BATCH_SIZE).prefetch(1)

if mode == 'm':
    p_model = tf.keras.applications.MobileNet(input_shape= (img_rows, img_cols,3))
    model = tf.keras.applications.MobileNet(input_tensor = p_model.input)
    model.load_weights("../../weights/fp_model_40_mobilenet.h5")
    p_model.load_weights("../../weights/p_model_40_mobilenet.h5")
    model.trainable = False
    p_model.trainable = False
    preprocess = tf.keras.applications.mobilenet.preprocess_input
    decode = tf.keras.applications.mobilenet.decode_predictions
    net = 'mobile'

elif mode == 'r':
    p_model = ResNet50(input_shape= (img_rows, img_cols,3))
    model = ResNet50(input_tensor = p_model.input)
    model.load_weights("../../weights/fp_model_40_resnet50.h5")
    p_model.load_weights("../../weights/p_model_40_resnet50.h5")
    model.trainable = False
    p_model.trainable = False
    preprocess = tf.keras.applications.resnet.preprocess_input
    decode = tf.keras.applications.resnet.decode_predictions
    net = 'res'

else:

    p_model = tf.keras.applications.DenseNet121(input_shape=(img_rows, img_cols,3))
    model = tf.keras.applications.DenseNet121(input_tensor = p_model.input)
    model.load_weights("../../weights/fp_model_40_densenet121.h5")
    p_model.load_weights("../../weights/p_model_40_densenet121.h5")
    model.trainable = False
    p_model.trainable = False
    preprocess = tf.keras.applications.densenet.preprocess_input
    decode = tf.keras.applications.densenet.decode_predictions
    net = 'dense'

model.compile()
p_model.compile()
    
def second(image,label):
    orig_image = tf.identity(image)
    input_image = tf.identity(image)
    orig_logist = tf.identity(model.predict(preprocess(input_image)[None,...]))
    orig_label =  np.argmax(orig_logist[0])

    
    quant_logist = tf.identity(p_model.predict(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    
    if orig_label != quant_label:
        return -2,-2,-2,-2,-2
    
    A = 0
    start_time = time.time()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            input_image_process = preprocess(input_image+A)[None,...]
            final_loss = loss_func(orig_label, p_model(input_image_process, training = False))



        grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = tf.clip_by_value(input_image + A, 0, 255)
        test_image = preprocess(test_image_deprocess)[None,...]
        pred1, pred2= model.predict(test_image), p_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        
        if not label1 == label2:
            if label1 == orig_label:
        
                total_time = time.time() - start_time
                
                norm = np.max(np.abs(A))
                
                return total_time, norm, iters, test_image_deprocess, A
            
    gen_img_deprocessed = tf.clip_by_value(orig_image + A, 0, 255)

    return -1, -1, -1, gen_img_deprocessed, A

def topk(model_pred, qmodel_pred, k):
    preds = decode(model_pred, top=k)
    qpreds = decode(qmodel_pred, top=1)[0][0][1]
    
    for pred in preds[0]:
        if pred[1] == qpreds:
            return True
    
    return False

def secondk(image,k):
    orig_image = tf.identity(image)
    input_image = tf.identity(image)
    orig_logist = tf.identity(model.predict(preprocess(input_image)[None,...]))
    orig_label =  np.argmax(orig_logist[0])

    
    quant_logist = tf.identity(p_model.predict(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])

    
    if orig_label != quant_label:
        return -2,-2,-2,-2,-2
    
    A = 0
    start_time = time.time()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    for iters in range(0,grad_iterations):
        with tf.GradientTape() as g:
            g.watch(input_image)
            input_image_process = preprocess(input_image+A)[None,...]
            final_loss = loss_func(orig_label, p_model(input_image_process, training = False))



        grads = normalize(g.gradient(final_loss, input_image))
        A += tf.sign(grads) * step
        A = tf.clip_by_value(A, -epsilon, epsilon)
        test_image_deprocess = tf.clip_by_value(input_image + A, 0, 255)
        test_image = preprocess(test_image_deprocess)[None,...]
        pred1, pred2= model.predict(test_image), p_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        
        if not topk(pred1, pred2, k):
            if label1 == orig_label:
        
                total_time = time.time() - start_time
                
                norm = np.max(np.abs(A))
                
                return total_time, norm, iters, test_image_deprocess, A
            
    gen_img_deprocessed = tf.clip_by_value(orig_image + A, 0, 255)

    return -1, -1, -1, gen_img_deprocessed, A

def calc_normal_success(method, methodk, ds, folderName='', filterName='',dataName='',dataFolder='',locald = ''):
    
    total=0
    count=0
    badimg = 0
    
    top5 = 0

    timeStore = []
    advdistStore = []
    stepsStore = []
    
    timeStorek = []
    advdistStorek = []
    stepsStorek = []
    failure = 0
    
    if not os.path.exists(locald):
        os.mkdir(locald)
        os.mkdir(locald+folderName)
        os.mkdir(locald+filterName)
        os.mkdir(locald+dataFolder)
                 
    if not os.path.exists(locald + 'failure/'):
        os.mkdir(locald + 'failure/')
        os.mkdir(locald + 'failure/' + folderName)
        os.mkdir(locald + 'failure/' + filterName)
    
    for i, features in enumerate(ds):

        images = features['image']
        labels = features['label']

        for j,image in enumerate(images):
            
            label = labels[j].numpy()

            time, advdist, steps, gen, A = method(image,label)

            total += 1

            if time == -1:
                print("Didnt find anything")
                np.save(locald + 'failure/' + folderName+"/"+dataName+str(failure)+"@"+str(total)+".npy", gen)
                np.save(locald + 'failure/' + filterName+"/"+dataName+str(failure)+"@"+str(total)+".npy", A)
                failure +=1
                continue
            
            if time == -2:
                badimg += 1
                total -= 1
                failure +=1
                print("Bad Image",badimg)
                continue
                
            if time == -3:
                badimg += 1
                total -= 1
                failure +=1
                print("Incorrect Image",badimg)
                continue

            count += 1
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
                
            print("starting k search")
            
            time, advdist, steps, gen, A = methodk(image,5)
            
            if time == -1:
                print("Didnt find anything in K")
                np.save(locald + 'failure/' + folderName+"/"+dataName+"k"+str(failure)+".npy", gen)
                np.save(locald + 'failure/' + filterName+"/"+ dataName+"k"+str(failure)+".npy", A)
                continue
            
            if time == -2:
                print("Bad Image in K",badimg)
                continue
            
            top5 += 1
            
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


calc_normal_success(second,secondk,mydataset,
                   folderName=net + 'net_imagenet_images_second', filterName=net +'net_imagenet_filters_second',dataName='second', dataFolder=net +'net_imagenet_data_second', locald ='../results/prune/PGD/'+net+'net/'  )
