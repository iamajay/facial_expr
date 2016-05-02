
from collections import Counter

import numpy as np
import theano
import theano.tensor as T

import network3
from network3 import sigmoid, tanh, ReLU, Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = net3.load_data_shared()
mini_batch_size = 1


def basic_conv(n=1, epochs=20):
    for j in range(n):
        print "Conv + FullyConnected architecture"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
        net.SGD(
            training_data, epochs, mini_batch_size, 0.1, validation_data, test_data)
    return net 

def dbl_conv(activation_fn=sigmoid):
    for j in range(1):
        print "Conv + Conv + FC architecture(double convolution)"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 256, 256), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 126, 126), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            ConvPoolLayer(image_shape=(mini_batch_size, 40, 61, 61), 
                          filter_shape=(40, 40, 6, 6), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            ConvPoolLayer(image_shape=(mini_batch_size, 40, 28, 28), 
                          filter_shape=(40, 40, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            ConvPoolLayer(image_shape=(mini_batch_size, 40, 12, 12), 
                          filter_shape=(40, 40, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=activation_fn),
            FullyConnectedLayer(
                n_in=40*4*4, n_out=500, activation_fn=activation_fn),
            SoftmaxLayer(n_in=500, n_out=7)], mini_batch_size)
        net.SGD(training_data, 10, mini_batch_size, 0.1, validation_data, test_data)
    return net 


def regularized_dbl_conv():
    for lmbda in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for j in range(1):
            print "Conv + Conv + FC num %s, with regularization %s" % (j, lmbda)
            net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                              filter_shape=(20, 1, 5, 5), 
                              poolsize=(2, 2)),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                              filter_shape=(40, 20, 5, 5), 
                              poolsize=(2, 2)),
                FullyConnectedLayer(n_in=40*4*4, n_out=100),
                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
            net.SGD(training_data, 20, mini_batch_size, 0.1, validation_data, test_data, lmbda=lmbda)


def double_fc_dropout(p0, p1, p2, repetitions):
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/face256.pkl.gz")
    nets = []
    for j in range(repetitions):
        print "\n\nTraining using a dropout network with parameters ",p0,p1,p2
        print "Training with expanded data, run num %s" % j
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 256, 256), 
                          filter_shape=(20, 1, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 126, 126), 
                          filter_shape=(40, 20, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 40, 61, 61), 
                          filter_shape=(80, 40, 6, 6), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 80, 28, 28), 
                          filter_shape=(100, 80, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 100, 12, 12), 
                          filter_shape=(160, 100, 5, 5), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            FullyConnectedLayer(
                n_in=160*4*4, n_out=1000, activation_fn=ReLU, p_dropout=p0),
            FullyConnectedLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=p1),
            SoftmaxLayer(n_in=1000, n_out=7, p_dropout=p2)], mini_batch_size)
        net.SGD(expanded_training_data, 10, mini_batch_size, 0.3, 
                validation_data, test_data)
        nets.append(net)
    return nets

def sample(p0,p1,p2,repetitions):
    expanded_training_data, _, _ = network3.load_data_shared(
        "../data/face256.pkl.gz")
    nets = []
    for j in range(repetitions):
        print "\n\nTraining using a dropout network with parameters ",p0,p1,p2
        print "Training with expanded data, run num %s" % j
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 256, 256), 
                          filter_shape=(20, 1, 3, 3), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 127, 127), 
                          filter_shape=(40, 20, 2, 2), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 40, 63, 63), 
                          filter_shape=(80, 40, 2, 2), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 80, 31, 31), 
                          filter_shape=(100, 80, 2, 2), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 100, 15, 15), 
                          filter_shape=(160, 100, 2, 2), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 160, 7, 7), 
                          filter_shape=(200, 160, 2, 2), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 200, 3, 3), 
                          filter_shape=(240, 200, 2, 2), 
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            FullyConnectedLayer(
                n_in=240*1*1, n_out=5000, activation_fn=ReLU, p_dropout=p0),
            FullyConnectedLayer(
                n_in=5000, n_out=2000, activation_fn=ReLU, p_dropout=p1),
            SoftmaxLayer(n_in=2000, n_out=7, p_dropout=p2)], mini_batch_size)
        net.SGD(expanded_training_data, 10, mini_batch_size, 0.3, 
                validation_data, test_data)
        nets.append(net)
    return nets


def run_experiments():

    nets = sample(0.5, 0.5, 0.5, 1)
    #dbl_conv()
    #double_fc_dropout(0.5,0.5,0.5,1)
    #basic_conv()
    #regularized_dbl_conv()


run_experiments()


