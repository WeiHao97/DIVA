import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import backend as K
import time

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# Quantization spec for Batchnormalization layer
class DefaultBNQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_activations):
        pass
    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

    def get_config(self):
        return {}
    
# Quantization spec (null) for concat layer    
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Use this config object if the layer has nothing to be quantized for 
    quantization aware training."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        # Does not quantize output, since we return an empty list.
        return []

    def get_config(self):
        return {}
    
# Quantization spec func for DenseNet
def apply_quantization(layer):
    if 'bn'  in layer.name:
        return tfmot.quantization.keras.quantize_annotate_layer(layer,DefaultBNQuantizeConfig())
    elif 'concat' in layer.name:
        return tfmot.quantization.keras.quantize_annotate_layer(layer,NoOpQuantizeConfig())
    else:
        return tfmot.quantization.keras.quantize_annotate_layer(layer)

# hyper-parameters    
BATCH_SIZE = 50
c = 1
grad_iterations = 20
step = 1
epsilon = 8
mode = 'm' # 'm' for MobileNet, 'r' for ResNet, 'd' for DenseNet
img_rows, img_cols, num_channel  = 224 ,224, 3 # input image dimensions

#Load Dataset
es = {'file_name': tf.TensorSpec(shape=(), dtype=tf.string, name=None),
 'image': tf.TensorSpec(shape=(img_rows, img_cols, num_channel), dtype=tf.float32, name=None),
 'label': tf.TensorSpec(shape=(), dtype=tf.int64, name=None)}
mydataset = tf.data.experimental.load("../../datasets/ImageNet/quantization/3kImages/",es).batch(BATCH_SIZE).prefetch(1)

# Construct models
if mode == 'm':
    model_ = tf.keras.applications.MobileNet(input_shape= (img_rows, img_cols,3))
    q_model = tfmot.quantization.keras.quantize_model(model_)
    model = tf.keras.applications.MobileNet(input_shape= (img_rows, img_cols,3))
    d_model = tf.keras.applications.MobileNet(input_tensor = q_model.input)
    model.load_weights("../../weights/fp_model_40_mobilenet.h5")# load model weight
    q_model.load_weights("../../weights/q_model_40_mobilenet.h5")
    d_model.load_weights("../../weights/distilled_fp_model_40_mobilenet.h5")
    model.trainable = False
    q_model.trainable = False
    d_model.trainable = False
    preprocess = tf.keras.applications.mobilenet.preprocess_input
    decode = tf.keras.applications.mobilenet.decode_predictions
    net = 'mobile'

elif mode == 'r':
    model_ = ResNet50(input_shape= (img_rows, img_cols,3))
    q_model = tfmot.quantization.keras.quantize_model(model_)
    model = ResNet50(input_shape= (img_rows, img_cols,3))
    d_model = ResNet50(input_tensor = q_model.input)
    model.load_weights("../../weights/fp_model_40_resnet50.h5")# load model weight
    q_model.load_weights("../../weights/q_model_40_resnet50.h5")
    d_model.load_weights("../../weights/distilled_fp_model_40_resnet50.h5")
    model.trainable = False
    q_model.trainable = False
    d_model.trainable = False
    preprocess = tf.keras.applications.resnet.preprocess_input
    decode = tf.keras.applications.resnet.decode_predictions
    net = 'res'

else:

    model_ = tf.keras.applications.DenseNet121(input_shape=(img_rows, img_cols,3))
    # Create a base model
    base_model = model_
    # Helper function uses `quantize_annotate_layer` to annotate that only the 
    # Dense layers should be quantized.

    LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
    MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer
    
    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` 
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_quantization,
    )

    with tfmot.quantization.keras.quantize_scope({'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig, 'NoOpQuantizeConfig': NoOpQuantizeConfig}):
        q_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    model = tf.keras.applications.DenseNet121(input_shape= (img_rows, img_cols,3))
    d_model = tf.keras.applications.DenseNet121(input_tensor = q_model.input)
    model.load_weights("../../weights/fp_model_40_densenet121.h5")# load model weight
    q_model.load_weights("../../weights/q_model_40_densenet121.h5")
    d_model.load_weights("../../weights/distilled_fp_model_40_densenet121.h5")
    model.trainable = False
    q_model.trainable = False
    d_model.trainable = False
    preprocess = tf.keras.applications.densenet.preprocess_input
    decode = tf.keras.applications.densenet.decode_predictions
    net = 'dense'



# DIVA attack for top-1
def second(image,label):
    orig_img = tf.identity(image)
    input_image = tf.identity(image)
    
    # Compute clean prediction and aquire labels
    orig_logist = tf.identity(model.predict(preprocess(input_image)[None,...]) )
    orig_label =  np.argmax(orig_logist[0])
    quant_logist = tf.identity(q_model.predict(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])
    d_logist =  tf.identity(d_model.predict(preprocess(input_image)[None,...]))
    d_label =  np.argmax(d_logist[0])

    # Check for unqualified input
    if orig_label != quant_label or orig_label != d_label:
        print(orig_label)
        return -2,-2,-2,-2,-2
    
    if orig_label != label:
        return -3,-3,-3,-3,-3
    
    # Initialize attack to 0
    A = 0
    start_time = time.time()
    
    for iters in range(0,grad_iterations):
        
        # Compute loss
        with tf.GradientTape() as g:
            g.watch(input_image)
            loss1 = K.mean(d_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            loss2 = K.mean(q_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            final_loss = K.mean(loss1 - c*loss2)

        # Compute attack
        grads = normalize(g.gradient(final_loss, input_image))
        adv_image = input_image + tf.sign(grads) * step
        A = tf.clip_by_value(adv_image - orig_img, -epsilon, epsilon)
        input_image = tf.clip_by_value(orig_img + A, 0, 255)
        test_image = preprocess(input_image)[None,...]
        
        # Compute new predictions
        pred1, pred2= d_model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        pred3 = model.predict(test_image)
        label3 = np.argmax(pred3[0])
        
        if not label1 == label2:
            if label1 == orig_label and decode(pred1, top=1)[0][0][2] > 0.6:
                # If successfully fool the quantized model but not the distilled fp model
                # also the conf score is higher than 0.6
            
                # time to generate the successful attack
                total_time = time.time() - start_time
                
                gen_img_deprocessed = input_image# adversarial image 
                orig_img_deprocessed = orig_img # original image
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy() # attack

                #Since the final goal for the attack is to keep undetected by the original model
                #its still a failure if the original model mispredicted the label
                if label3 != orig_label:
                    return -1, -1, -1, gen_img_deprocessed, A
                
                norm = np.max(np.abs(A)) # adversarial distance
                
                return total_time, norm, iters, gen_img_deprocessed, A

    gen_img_deprocessed = input_image # generated non-adversarial image
    orig_img_deprocessed = orig_img # original image
    A = (gen_img_deprocessed - orig_img_deprocessed).numpy() # differences

    return -1, -1, -1, gen_img_deprocessed, A

# Top-k evaluation
def topk(model_pred, qmodel_pred, k):
    preds = decode(model_pred, top=k)
    qpreds = decode(qmodel_pred, top=1)[0][0][1]
    
    for pred in preds[0]:
        if pred[1] == qpreds:
            return True
    
    return False

# DIVA attack for top-k
def secondk(image,k):
    orig_img = tf.identity(image)
    input_image = tf.identity(image)
    
    # Compute clean prediction and aquire labels
    orig_logist = tf.identity(model.predict(preprocess(input_image)[None,...]) )
    orig_label =  np.argmax(orig_logist[0])
    quant_logist = tf.identity(q_model.predict(preprocess(input_image)[None,...]))
    quant_label =  np.argmax(quant_logist[0])
    d_logist =  tf.identity(d_model.predict(preprocess(input_image)[None,...]))
    d_label =  np.argmax(d_logist[0])

    # Check for unqualified input
    if orig_label != quant_label or orig_label != d_label:
        return -2,-2,-2,-2,-2
    
    # Initialize attack to 0
    A = 0
    start_time = time.time()
    
    for iters in range(0,grad_iterations):
        
        # Compute loss
        with tf.GradientTape() as g:
            g.watch(input_image)
            loss1 = K.mean(d_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            loss2 = K.mean(q_model(preprocess(input_image)[None,...], training = False)[..., orig_label])
            final_loss = K.mean(loss1 - c*loss2)

        # Compute attack
        grads = normalize(g.gradient(final_loss, input_image))
        adv_image = input_image + tf.sign(grads) * step
        A = tf.clip_by_value(adv_image - orig_img, -epsilon, epsilon)
        input_image = tf.clip_by_value(orig_img + A, 0, 255)
        test_image = preprocess(input_image)[None,...]
        
        # Compute new predictions
        pred1, pred2= d_model.predict(test_image), q_model.predict(test_image)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
        pred3 = model.predict(test_image)
        label3 = np.argmax(pred3[0])
        
        if not topk(pred1, pred2, k):
            if label1 == orig_label and decode(pred1, top=1)[0][0][2] > 0.6:
                # If successfully fool the quantized model but not the distilled fp model
                # also the conf score is higher than 0.6
            
                # time to generate the successful attack
                total_time = time.time() - start_time
                gen_img_deprocessed = input_image# adversarial image
                orig_img_deprocessed = orig_img # original image
                A = (gen_img_deprocessed - orig_img_deprocessed).numpy()# attack 
                    
                #Since the final goal for the attack is to keep undetected by the original model
                #its still a failure if the original model mispredicted the label
                if label3 == orig_label and not topk(pred3, pred2, k):
                    norm = np.max(np.abs(A))# adversarial distance
                    return total_time, norm, iters, gen_img_deprocessed, A

                else:
                    return -1, -1, -1, gen_img_deprocessed, A
    
    gen_img_deprocessed = input_image# generated non-adversarial image
    orig_img_deprocessed = orig_img# original image
    A = (gen_img_deprocessed - orig_img_deprocessed).numpy()# differences

    return -1, -1, -1, gen_img_deprocessed, A

def calc_normal_success(method, methodk, ds, folderName='', filterName='',dataName='',dataFolder='',locald = ''):
    
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
    
    for i, features in enumerate(ds):

        images = features['image']
        labels = features['label']

        for j,image in enumerate(images):
            
            label = labels[j].numpy()

            # attampt for the top-1 attack
            time, advdist, steps, gen, A = method(image,label)

            total += 1

            # if attack failed
            if time == -1:
                print("Didnt find anything")
#                 np.save(locald + 'failure/' + folderName+"/"+dataName+str(failure)+"@"+str(total)+".npy", gen)
#                 np.save(locald + 'failure/' + filterName+"/"+dataName+str(failure)+"@"+str(total)+".npy", A)
                failure +=1
                continue
            
            # if its a bad image
            if time == -2:
                badimg += 1
                total -= 1
                failure +=1
                print("Bad Image",badimg)
                continue
             
            # if its an incorrect image
            if time == -3:
                badimg += 1
                total -= 1
                failure +=1
                print("Incorrect Image",badimg)
                continue

            count += 1 # top-1 sucecced
#             np.save(locald+folderName+"/"+dataName+str(count)+"@"+str(total)+".npy", gen)
#             np.save(locald+filterName+"/"+dataName+str(count)+"@"+str(total)+".npy", A)
            
            print("Number seen:",total)
            print("No. worked:", count)
            print("No. topk:", top5)
            print("Bad Image:", badimg)
            
            timeStore.append(time)
            advdistStore.append(advdist)
            stepsStore.append(steps)
            
#             with open(locald+dataFolder+"/"+dataName+'_time_data.csv', 'a') as f:
#                 f.write(str(time) + ", ")

#             with open(locald+dataFolder+"/"+dataName+'_advdist_data.csv', 'a') as f:
#                 f.write(str(advdist) + ", ")
            
#             with open(locald+dataFolder+"/"+dataName+'_steps_data.csv', 'a') as f:
#                 f.write(str(steps) + ", ")
            
            # attampt for the top-5 attack
            print("starting k search")
            
            time, advdist, steps, gen, A = methodk(image,5)
            
            # if attack failed
            if time == -1:
                print("Didnt find anything in K")
                #np.save(locald + 'failure/' + folderName+"/"+dataName+"k"+str(failure)+".npy", gen)
                #np.save(locald + 'failure/' + filterName+"/"+ dataName+"k"+str(failure)+".npy", A)
                continue
            
            # if its a bad image
            if time == -2:
                print("Bad Image in K",badimg)
                continue
            
            top5 += 1
            
            #np.save(locald+folderName+"/"+dataName+"k"+str(count)+".npy", gen)
            #np.save(locald+filterName+"/"+dataName+"k"+str(count)+".npy", A)
            
            timeStorek.append(time)
            advdistStorek.append(advdist)
            stepsStorek.append(steps)
        
            #with open(locald+dataFolder+"/"+dataName+'_timek_data.csv', 'a') as f:
                #f.write(str(time) + ", ")

            #with open(locald+dataFolder+"/"+dataName+'_advdistk_data.csv', 'a') as f:
                #f.write(str(advdist) + ", ")
            
            #with open(locald+dataFolder+"/"+dataName+'_stepsk_data.csv', 'a') as f:
                #f.write(str(steps) + ", ")

    print("Number seen:",total)
    print("No. worked:", count)
    print("No. topk:", top5)
    print("Bad Image:", badimg)



calc_normal_success(second,secondk,mydataset,
                   folderName=net + 'net_imagenet_images_second', filterName=net +'net_imagenet_filters_second',dataName='second', dataFolder=net +'net_imagenet_data_second', locald ='./results/SemiBB/' + net + 'net_c1/'+ net + 'net/')