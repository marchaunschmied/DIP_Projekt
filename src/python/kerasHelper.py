#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:16:34 2019

@author: marsu
version 3.0

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
    

def plotAcc(history, smooth=False):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    if smooth:    
        plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
        plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
    else:
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

def plotLoss(history, smooth=False): 
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    if smooth: 
        plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
        plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')       
    else:
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    #todo show only integer epochs
    
def gradCamVGG16(img, preprocessedImg, model):

    preds = model.predict(preprocessedImg)
    #print('Predicted:', decode_predictions(preds,top=3)[0])

    classNr=np.argmax(preds[0])
    
    model_output= model.output[:,classNr] #softmax
    last_conv_layer = model.get_layer('block5_conv3')
    
    from keras import backend as K
    grads = K.gradients(model_output, last_conv_layer.output)[0]
    
    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([preprocessedImg])
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # We use cv2 to load the original image
    #img = cv2.imread(img_path)
    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap * 0.4 + img
    # Save the image to disk
    return superimposed_img.astype('uint')

def freeze_session(model, filename):
    # adapted from https://github.com/opencv/opencv_contrib/issues/1241
    #              https://answers.opencv.org/question/183682/opencv-dnn-import-dropout-layer-error-after-finetuning-keras-vgg16/

    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    output_names=[out.op.name for out in model.outputs]  
    clear_devices=True   
    keep_var_names=False
    K.set_learning_phase(0)
    session=K.get_session()
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        #return frozen_graph
    tf.train.write_graph(frozen_graph, "./", filename, as_text=False)

    


