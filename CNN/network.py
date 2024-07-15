'''
Description: methods to set up and train the network's parameters.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from CNN.forward import *
from CNN.backward import *
from CNN.utils import *

import numpy as np
import pickle
from tqdm import tqdm
#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, conv_s, pool_f, pool_s):

    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 # pass through ReLU non-linearity

    conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity

    pooled = avgpool(conv2, pool_f, pool_s) # maxpooling operation

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer

    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity

    out = w4.dot(z) + b4 # second dense layer

    probs = softmax(out) # predict class probabilities with the softmax activation function

    ################################################
    #################### Loss ######################
    ################################################

    loss = categoricalCrossEntropy(probs, label) # categorical cross-entropy loss

    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label # derivative of loss w.r.t. final dense layer output
    dw4 = dout.dot(z.T) # loss gradient of final dense layer weights
    db4 = np.sum(dout, axis = 1).reshape(b4.shape) # loss gradient of final dense layer biases

    dz = w4.T.dot(dout) # loss gradient of first dense layer outputs
    dz[z<=0] = 0 # backpropagate through ReLU
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis = 1).reshape(b3.shape)

    dfc = w3.T.dot(dz) # loss gradients of fully-connected layer (pooling layer)
    dpool = dfc.reshape(pooled.shape) # reshape fully connected into dimensions of pooling layer

    dconv2 = meanpoolBackward(dpool, conv2, pool_f, pool_s) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv2[conv2<=0] = 0 # backpropagate through ReLU

    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_s) # backpropagate previous gradient through second convolutional layer.
    dconv1[conv1<=0] = 0 # backpropagate through ReLU

    dimage, df1, db1 = convolutionBackward(dconv1, image, f1, conv_s) # backpropagate previous gradient through first convolutional layer.

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss

#####################################################
################### Optimization ####################
#####################################################

import numpy as np

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    Update the parameters through Adam gradient descent.
    '''
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    X = batch[:, 0:-1].reshape(len(batch), n_c, dim, dim)
    Y = batch[:, -1]
    batch_size = len(batch)

    grads = {name: np.zeros(param.shape) for name, param in zip(['f1', 'f2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4'], params)}
    vs = {name: np.zeros(param.shape) for name, param in zip(['f1', 'f2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4'], params)}
    ss = {name: np.zeros(param.shape) for name, param in zip(['f1', 'f2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4'], params)}
    cost_ = 0

    for i in range(batch_size):
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)
        example_grads, loss = conv(x, y, params, 1, 2, 2)

        for (name, g), grad in zip(grads.items(), example_grads):
            g += grad
        cost_ += loss

    # Parameter update
    for name, param in zip(['f1', 'f2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4'], params):
        g = grads[name]
        v = vs[name]
        s = ss[name]

        v = beta1 * v + (1 - beta1) * g / batch_size
        s = beta2 * s + (1 - beta2) * (g / batch_size) ** 2
        param -= lr * v / (np.sqrt(s) + 1e-7)

    cost.append(cost_ / batch_size)
    return params, cost



#用sgd算法出现了过拟合的现象，导致正确率显著下降，训练一次是30，训练两次降低到了13
# def sgd(batch, num_classes, lr, dim, n_c, params, cost):
#     '''
#     Update the parameters through Stochastic Gradient Descent (SGD).
#     '''
#     [f1, f2, w3, w4, b1, b2, b3, b4] = params
#
#     X = batch[:, 0:-1]  # get batch inputs
#     X = X.reshape(len(batch), n_c, dim, dim)
#     Y = batch[:, -1]  # get batch labels
#
#     cost_ = 0
#     batch_size = len(batch)
#
#     # Initialize gradients
#     df1 = np.zeros(f1.shape)
#     df2 = np.zeros(f2.shape)
#     dw3 = np.zeros(w3.shape)
#     dw4 = np.zeros(w4.shape)
#     db1 = np.zeros(b1.shape)
#     db2 = np.zeros(b2.shape)
#     db3 = np.zeros(b3.shape)
#     db4 = np.zeros(b4.shape)
#
#     for i in range(batch_size):
#         x = X[i]
#         y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot
#
#         # Collect gradients for training example
#         grads, loss = conv(x, y, params, 1, 2, 2)
#         [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
#
#         df1 += df1_
#         db1 += db1_
#         df2 += df2_
#         db2 += db2_
#         dw3 += dw3_
#         db3 += db3_
#         dw4 += dw4_
#         db4 += db4_
#
#         cost_ += loss
#
#     # Parameter update using SGD
#     f1 -= lr * df1 / batch_size
#     b1 -= lr * db1 / batch_size
#     f2 -= lr * df2 / batch_size
#     b2 -= lr * db2 / batch_size
#     w3 -= lr * dw3 / batch_size
#     b3 -= lr * db3 / batch_size
#     w4 -= lr * dw4 / batch_size
#     b4 -= lr * db4 / batch_size
#
#     cost_ = cost_ / batch_size
#     cost.append(cost_)
#
#     params = [f1, f2, w3, w4, b1, b2, b3, b4]
#
#     return params, cost


#####################################################
##################### Training ######################
#####################################################

def train(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 7, num_filt1 = 1, num_filt2 = 1, batch_size = 32, num_epochs = 1):

    # training data
    m = 60000
    X = extract_data('./MNIST/raw/train-images-idx3-ubyte')
    y_dash = extract_labels('./MNIST/raw/train-labels-idx1-ubyte').reshape(m, 1)
    X -= int(np.mean(X))
    X /= int(np.std(X))
    train_data = np.hstack((X, y_dash))

    np.random.shuffle(train_data)

    # Initializing all the parameters
    f1, f2, w3, w4 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (16, 64), (10, 16)
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeWeight(w3)
    w4 = initializeWeight(w4)

    b1 = np.zeros((f1.shape[0], 1))
    b2 = np.zeros((f2.shape[0], 1))
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    cost = []

    print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))

    return params, cost