
#Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
from sklearn.metrics import accuracy_score

sess = tf.Session()
saver = tf.train.import_meta_graph('/home/user/Desktop/MyTensorBoard/MNIST VJ/saved_model/mnist_model.ckpt.meta')
saver.restore(sess, '/home/user/Desktop/MyTensorBoard/MNIST VJ/saved_model/mnist_model.ckpt')
graph = tf.get_default_graph()

y = graph.get_tensor_by_name('OUTPUT:0')
x =  graph.get_tensor_by_name('INPUT:0')

list=['zero','one','two','three','four','five','six','seven','eight', 'nine']

image=cv2.imread('/home/user/Downloads/number0_new.png', 0)
image=cv2.resize(image,(28,28))
image=image.flatten()
image =image.reshape(1, 784)

predictions=sess.run(y, feed_dict={x:image})

predictions=predictions[0].tolist()
print predictions
print("Predicted Value==>",list[predictions.index(1)])