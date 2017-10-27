import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import  cv2
from tf_utils import  random_mini_batches, predict
from pickle_utils import save_obj


hdf5_path = 'train_hdf5/dataset.hdf5'

hdf5_file = h5py.File(hdf5_path, "r")

X_train_orig = hdf5_file["X_train"]
Y_train_orig = hdf5_file["Y_train"]
Y_train_orig = np.reshape(Y_train_orig, (1, len(Y_train_orig)))

print(X_train_orig.shape)
print(Y_train_orig.shape)
print(Y_train_orig[0][1])

plt.ion()

#plt.imshow(cv2.cvtColor(np.reshape(X_train_orig[:,1].T, (64,64,3)), cv2.COLOR_BGR2RGB))
#plt.pause(0.001)
#input("Press [enter] to continue.")

def init_parameters(n_input, n_units_in_layers):
    parameters = {}
    for i in range(len(n_units_in_layers)):
        if(i==0):
            W = tf.get_variable("W"+str(i+1), [n_units_in_layers[i], n_input], initializer = tf.contrib.layers.xavier_initializer())
        else:
            W = tf.get_variable("W"+str(i+1), [n_units_in_layers[i], n_units_in_layers[i-1]], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b"+str(i+1) , [n_units_in_layers[i],1], initializer = tf.zeros_initializer())
        parameters["W"+str(i+1)] = W
        parameters["b"+str(i+1)] = b
        
    return parameters
    
def forward_propagation(X, parameters):
    A_prev = X
    for i in range(len(parameters) // 2 - 1):
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
    
    
def nn_model(X_train, Y_train, n_units_in_layers, learning_rate = 0.0001 , num_epochs = 1500, minibatch_size = 32, print_cost = True):
   
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    print("examples: ", m)
    
    X = tf.placeholder(tf.float32, [X_train.shape[0], None], name = "input")
    Y = tf.placeholder(tf.float32, [Y_train.shape[0], None])
    
    parameters = init_parameters(n_x, n_units_in_layers)
    
    Z = forward_propagation(X, parameters)
    
    cost = compute_cost(Z, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):
        
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
               
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True: # and epoch % 5 == 0:
                costs.append(epoch_cost)
                
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        plt.pause(20)
        plt.savefig("cost.png")
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        saver.save(sess, 'model/model.ckpt')

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.greater_equal(tf.nn.sigmoid(Z), 0.5), tf.equal(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
        
        
        
        
        
        
X_train = np.array(X_train_orig) / 255
Y_train = Y_train_orig

# LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
n_units_in_layers = [500, 500,1]

learning_rate = 0.0001


num_epochs = 150
minibatch_size = 64

params = nn_model(X_train, Y_train, n_units_in_layers, learning_rate, num_epochs, minibatch_size, True)
save_obj(params, "params_0001")



























