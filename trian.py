
#Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
from sklearn.metrics import accuracy_score

#set random seed
tf.set_random_seed(79)
np.random.seed(79)

#Global Variables
learning_rate=0.001
batch_size=20
epoch=250

#Pathss
train_data_path="/home/user/Desktop/MyTensorBoard/MNIST VJ/mnist_png/train/"
test_data_path="/home/user/Desktop/MyTensorBoard/MNIST VJ/mnist_png/testing/"

#subdirectories
train_data=os.listdir(train_data_path)
test_data=os.listdir(test_data_path)

#variables to store train and test data
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]


#train data
print("Train Data")
for i in range(len(train_data)):
	#store subdirectories path
	current_directories=train_data_path+train_data[i]+"/"
	#files of subdirectories
	files=os.listdir(current_directories)
	for j in range(len(files)):
		image=cv2.imread(current_directories+files[j], 0)
		image=image.flatten()
		X_train.append(image)
		Y_train.append(int(train_data[i]))
		#if(j==1000):
		#	break
	print("Done with " + str(i) + " out of " + str(len(train_data)))


#train dataframe
train_df=pd.DataFrame(X_train)
train_df['labels']=Y_train
train_df=train_df.sample(frac=1.0).reset_index(drop=True)
X_train=train_df.iloc[:,  :784]
Y_train=train_df['labels']

print train_df

#test data
print("Test Data")
for i in range(len(train_data)):
	#store subdirectories path
	current_directories=test_data_path+test_data[i]+"/"
	#files of subdirectories
	files=os.listdir(current_directories)
	for j in range(len(files)):
		image=cv2.imread(current_directories+files[j], 0)
		image=image.flatten()
		X_test.append(image)
		Y_test.append(int(test_data[i]))
		if(j==1000):
			break
	print("Done with " + str(i) + " out of " + str(len(test_data)))
	
#test dataframe
test_df=pd.DataFrame(X_test)
test_df['labels']=Y_test
test_df=test_df.sample(frac=1.0).reset_index(drop=True)
X_test=test_df.iloc[:,  :784]
Y_test=test_df['labels']


#one hot vector
Y_train_one_hot=pd.get_dummies(Y_train)
Y_test_one_hot=pd.get_dummies(Y_test)

print Y_test_one_hot
print Y_train_one_hot

#print shapes of data
print("Train Data ", X_train.shape, Y_train.shape, Y_train_one_hot.shape)
print("Test Data ", X_test.shape, Y_test.shape, Y_test_one_hot.shape)

#placeholder
X=tf.placeholder(tf.float32, [None, 784], name='INPUT')
Y=tf.placeholder(tf.float32, [None, 10])

def build_network():
	# first layer
	weights_1 = tf.Variable(tf.random_normal([784,  256]))
	bias_1 = tf.Variable(tf.random_normal([256]))
	layer_1 = tf.add(tf.matmul(X,  weights_1),  bias_1)

	# second layer
	weights_2 = tf.Variable(tf.random_normal([256,  256]))
	bias_2 = tf.Variable(tf.random_normal([256]))
	layer_2 = tf.add(tf.matmul(layer_1,  weights_2),  bias_2)

	# third layer
	weights_3 = tf.Variable(tf.random_normal([256,  256]))
	bias_3 = tf.Variable(tf.random_normal([256]))
	layer_3 = tf.add(tf.matmul(layer_2,  weights_3),  bias_3)


	# fourth layer
	weights_4 = tf.Variable(tf.random_normal([256,  256]))
	bias_4 = tf.Variable(tf.random_normal([256]))
	layer_4 = tf.add(tf.matmul(layer_3,  weights_4),  bias_4)

	# output layer
	weights_out = tf.Variable(tf.random_normal([256,  10]))
	bias_out = tf.Variable(tf.random_normal([10]))
	out_layer = tf.add(tf.matmul(layer_4,  weights_out),  bias_out)
	return out_layer
	
# prediction-probabilities
logits=build_network()
pred_prob = tf.nn.softmax(logits, name= 'OUTPUT')

# get prediction
predict = tf.argmax(pred_prob,  1)

# loss or cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,  labels=Y))

# optimizer function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize variables
init = tf.global_variables_initializer()

# make session and initialize it
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
save_path="/home/user/Desktop/MyTensorBoard/MNIST VJ/saved_model/mnist_model.ckpt"

print("Training network")
for e in range(epoch):
	strt_pt=0
	end_pt=batch_size
	print("=======Epoch  "+str(e)+" Out of  "+str(epoch)+"=========")
	while(end_pt<=len(X_train)):
		X_batch=X_train.iloc[strt_pt:end_pt,  :]
		Y_batch_one_hot=Y_train_one_hot.iloc[strt_pt:end_pt,  :]
		#train network
		sess.run(optimizer, feed_dict={X:X_batch, Y:Y_batch_one_hot})
		strt_pt=end_pt
		end_pt += batch_size
		if(strt_pt<len(X_train) and end_pt<len(X_train)):
			end_pt=len(X_train)
	#train loss,  accuracy
	loss, pred=sess.run([cost, predict], feed_dict={X:X_train, Y:Y_train_one_hot})
	#accuracy
	acc=accuracy_score(pred, Y_train)
	#loss and accuracy of train data
	print("Train loss=  {} and accuracy= {}".format(loss, acc))
	#test loss and accuracy
	loss, pred=sess.run([cost, predict], feed_dict={X:X_test, Y:Y_test_one_hot})
	#test data accuracy
	acc=accuracy_score(pred, Y_test)
	#loss and accuracy of test data
	print("Test loss=  {} and accuracy= {}\n".format(loss, acc))

save_path = saver.save(sess, save_path)
print("Model Saved==>",save_path)

