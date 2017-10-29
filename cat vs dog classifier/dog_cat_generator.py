import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pickle_utils import load_obj
import scipy
from scipy import ndimage
import cv2
import matplotlib
from predictor import forward_propagation_np


def init_parameters(n_input, n_units_in_layers):
    parameters = {}
    for i in range(len(n_units_in_layers)):
        if(i==0):
            W = tf.placeholder(tf.float32, shape=(n_units_in_layers[i], n_input))
        else:
            W = tf.placeholder(tf.float32, shape=(n_units_in_layers[i], n_units_in_layers[i-1]))
        b = tf.placeholder(tf.float32, shape=(n_units_in_layers[i], 1))
        parameters["W"+str(i+1)] = W
        parameters["b"+str(i+1)] = b

    X = tf.Variable(tf.ones((n_input, 1)))
        
    return X, parameters
    
def get_feed_dict(params_tf, params_orig):
    feed_dict = {}
    for key,value in params_tf.items():
        print(key, ": ", np.shape(params_orig[key]))
        feed_dict[value] = params_orig[key]
    return feed_dict
    

def forward_propagation(X, parameters):
    A_prev = X
    for i in range(len(parameters) // 2 - 1):
        print("i+1:", i+1)
        print(parameters["W"+str(2)])
        print(A_prev)
        Z = tf.matmul(parameters["W"+str(i+1)], A_prev) + parameters["b"+str(i+1)]
        A = tf.nn.relu(Z)
        A_prev = A
    i+=1
    Z = tf.add(tf.matmul(parameters["W"+str(i+1)], A_prev), parameters["b"+str(i+1)], name="y_pred")
    return Z
    
    
def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost
    
    
def nn_model(n_x, params_orig, Y, n_units_in_layers, learning_rate = 0.0001 , num_epochs = 10, print_cost = True):
   
    m=1
    n_y = 1
    costs = []
    
    X, parameters = init_parameters(n_x, n_units_in_layers)
    
    Z = forward_propagation(X, parameters)
    
    cost = compute_cost(Z, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    img = []
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        feed_dict = get_feed_dict(parameters, params_orig)
        
        for epoch in range(num_epochs):
            _ , epoch_cost = sess.run([optimizer, cost], feed_dict=feed_dict)
            if(epoch%100 == 0):
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if(epoch%5 == 0):
                costs.append(epoch_cost)
        
        plt.ion()
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        plt.pause(0.001)
        plt.savefig("generator_cost.png")
        
        img = sess.run(X)
        
        print("prediction: ", forward_propagation_np(img, params_orig))
        
        img = cv2.cvtColor(np.reshape(img, (64,64,3)), cv2.COLOR_BGR2RGB)
        img = scipy.misc.imresize(np.reshape(img, (64,64,3)), (256,256,3))
        matplotlib.image.imsave('generated_cat.jpg', img)
        print("output stored to generated_cat.jpg")
        
    return img
            


            
n_x = 64*64*3
params_orig = load_obj("params_0001")
Y = np.array([0.90540236], dtype="float32").reshape((1,1))
n_units_in_layers = [500,500,1]

image = nn_model(n_x, params_orig, Y, n_units_in_layers,learning_rate = 0.000001, num_epochs = 5000)
print("yo")
'''
plt.close()
image = cv2.cvtColor(np.reshape(image, (64,64,3)), cv2.COLOR_BGR2RGB);
plt.imshow(image)
matplotlib.image.imsave('generated_cat.jpg', scipy.misc.imresize(np.reshape(image, (64,64,3)), (256,256,3)))
plt.pause(5)
'''












