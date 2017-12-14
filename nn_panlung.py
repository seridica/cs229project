# -*- coding: utf-8 -*-
"""
Neural Network for Pan-Lung Data
November 30 2017
CS229 Project

A neural network for the pan-lung data based on the one developed for 
homework 4

There is one hidden layers.

The output layer has either two nodes (representing presence or absence
of metastasis), or a node for each tumor class.

Cost function in binary case is log-loss, cost function in multi-class
case is softmax cross-entropy.

@author: Calvin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from random import *
import io
import sys
import pickle
from count_features import *
from generate_labels import *
from feature_clean import *
import pdb

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
    ### YOUR CODE HERE
    # Trick for dealing with overflow / underflow
    n_samples, n_nodes = x.shape
    z = x - ( np.matlib.repmat(np.max(x, axis=1), n_nodes, 1) ).T
    s = np.divide( np.exp(z), ( np.matlib.repmat(np.sum( np.exp(z),axis=1), n_nodes, 1) ).T )
    ### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s = np.divide(1, 1+np.exp(-x))
    ### END YOUR CODE
    return s

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    # Useful size information
    (data_size, num_hidden) = W1.shape
    num_samples = len(labels)
    
    # Vectorize b1 - note, input data and weights are already vectorized
    # B1 = (num_samples x num_hidden)
    B1_vectorized = np.matlib.repmat(b1, num_samples, 1)
    
    # Solve for hidden layer z
    # z1 = (num_samples x num_hidden)
    z1 = data.dot(W1) + B1_vectorized
    
    # Activation from hidden layer
    # h = (num_samples x num_hidden)
    h = sigmoid(z1)
    
    # Vectorize b2 - note, hidden layer activation and weights are already
    # properly vectorized
    # B2 = (num_samples x data_size)
    B2_vectorized = np.matlib.repmat(b2, num_samples, 1)
    
    # Solve for output layer z2
    # z2 = (num_samples x data_size)
    z2 = h.dot(W2) + B2_vectorized
    
    # Output layer
    # y = (num_samples x data_size)
    y = softmax(z2)
    
    # Compute cross-entropy cost
    cost = -np.sum( np.multiply(labels, np.log(y)) ) / num_samples
    
    ### END YOUR CODE
    return h, y, cost

def backward_prop(data, labels, params, lam):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    # Repeat forward propagation code for various intermediate values
    # Useful size information
    (data_size, num_hidden) = W1.shape
    num_samples = len(labels)
    
    # Vectorize b1 - note, input data and weights are already vectorized
    # B1 = (num_samples x num_hidden)
    B1_vectorized = np.matlib.repmat(b1, num_samples, 1)
    
    # Solve for hidden layer z
    # z1 = (num_samples x num_hidden)
    z1 = data.dot(W1) + B1_vectorized
    
    # Activation from hidden layer
    # h = (num_samples x num_hidden)\
    h = sigmoid(z1)
    
    # Vectorize b2 - note, hidden layer activation and weights are already
    # properly vectorized
    # B2 = (num_samples x data_size)
    B2_vectorized = np.matlib.repmat(b2, num_samples, 1)
    
    # Solve for output layer z2
    # z2 = (num_samples x data_size)
    z2 = h.dot(W2) + B2_vectorized
    
    # Partial derivative of CE w.r.t. z[2]
    part_z2_ce = -( labels - softmax(z2) ) / num_samples
    
    # Gradient W2
    gradW2 = (h.T).dot(part_z2_ce) + lam*W2
    
    # Gradient b2
    sample_ones = np.ones(num_samples)
    gradb2 = sample_ones.dot(part_z2_ce)
    
    # Gradient W1
    sigm_temp = np.multiply( h, 1 - h )
    part_z1_ce = np.multiply(sigm_temp, (part_z2_ce.dot(W2.T)))
    gradW1 = (data.T).dot(part_z1_ce) + lam*W1
    
    # Gradient b1
    gradb1 = sample_ones.dot(part_z1_ce)

    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2
    
    return grad

def nn_train(trainData, trainLabels, devData, devLabels, lam):
    (m, n) = trainData.shape
    num_hidden = 15
    learning_rate = 0.1
    params = {}

    ### YOUR CODE HERE
    # Initialize parameters
    K = trainLabels.shape[1] # Number of output nodes (classes)
    params['W1'] = np.random.randn(n, num_hidden)
    params['b1'] = np.zeros(num_hidden)
    params['W2'] = np.random.randn(num_hidden, K)
    params['b2'] = np.zeros(K)
    
    # Minibatch size
    B = 50
    num_batches = int( np.floor(m/B) )
    
    max_epochs = 300
    cost_trend = np.zeros((max_epochs,2))
    accuracy_trend = np.zeros((max_epochs,2))
    epoch_list = np.linspace(1,max_epochs,max_epochs)
    for i in range(max_epochs):
        #print("\nEpoch: ")
        #print(i)
        for j in range(num_batches):
            #batchTrainData = trainData[(j*B):((j+1)*B), :]
            #batchLabels = trainLabels[(j*B):((j+1)*B), :]
            if (j==1-num_batches):
                batchTrainData = trainData[(j*B):,:]
                batchLabels = trainLabels[(j*B):,:]
            else:
                batchTrainData = trainData[(j*B):((j+1)*B),:]
                batchLabels = trainLabels[(j*B):((j+1)*B),:]
            
            grad = backward_prop(batchTrainData, batchLabels, params, lam)
            
            params['W1'] = params['W1'] - learning_rate * grad['W1']
            params['W2'] = params['W2'] - learning_rate * grad['W2']
            params['b1'] = params['b1'] - learning_rate * grad['b1']
            params['b2'] = params['b2'] - learning_rate * grad['b2']
    
        hTrain, outputTrain, costTrain = forward_prop(trainData, trainLabels, params)
        accuracyTrain = compute_accuracy(outputTrain, trainLabels)
        
        hDev, outputDev, costDev = forward_prop(devData, devLabels, params)
        accuracyDev = compute_accuracy(outputDev, devLabels)
        
        accuracy_trend[i,0] = accuracyTrain
        accuracy_trend[i,1] = accuracyDev
        
        cost_trend[i,0] = costTrain
        cost_trend[i,1] = costDev
        
        #print(accuracyTrain)
        #print(accuracyDev)
#        
#    fig1 = plt.figure()
#    plt.hold(True)
#    plt.plot(epoch_list, accuracy_trend[:,0], label="Train")
#    plt.plot(epoch_list, accuracy_trend[:,1], label="Dev")
#    plt.xlabel('Epoch')
#    plt.ylabel('Accuracy')
#    plt.legend()
#    plt.title('Neural Network Training Accuracy')
#    #fig1.savefig('NN_acc_reg.png')
#    
#    fig2 = plt.figure()
#    plt.hold(True)
#    plt.plot(epoch_list, cost_trend[:,0], label="Train")
#    plt.plot(epoch_list, cost_trend[:,1], label="Dev")
#    plt.xlabel('Epoch')
#    plt.ylabel('Cost')
#    plt.legend()
#    plt.title('Neural Network Training Cost')
#    #fig2.savefig('NN_cost_reg.png')
#    ### END YOUR CODE

    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, int(np.max(labels))+1))
    for i in range(labels.size):
        one_hot_labels[i,int(labels[i])] = 1
    #one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    print('Loading Data')
    #trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    
    ## File paths etc.
    dataPath = 'D:\Stanford\CS229\Project\Data'
    
    # Load data
    tcgaDict = pickle.load(open(dataPath + "\\tcgaData.pickle", "rb"))
    luadDict = pickle.load(open(dataPath + "\luadData.pickle", "rb"))
    
    """
    Build feature matrix - Change the function to use a different feature set
    """
    ## Special feature matrix of all features combined
    #feat1 = num_mutation_type(tcgaDict)
    #feat1Luad = num_mutation_type(luadDict)
    #feat2 = num_mutations_per_gene(tcgaDict)
    #feat2Luad = num_mutations_per_gene(luadDict)
    #feat4 = num_mutation_eff(tcgaDict)
    #feat4Luad = num_mutation_eff(luadDict)
    #feat5 = num_SNP(tcgaDict)
    #feat5Luad = num_SNP(luadDict)
    #rawFeatures = np.concatenate((feat1, feat2, feat4, feat5), axis=1)
    #rawFeaturesLuad = np.concatenate((feat1Luad, feat2Luad, feat4Luad, feat5Luad), axis=1)
    
    #rawFeatures = num_mutations_per_gene(tcgaDict)
    #rawFeaturesLuad = num_mutations_per_gene(luadDict)
    
    #rawFeatures = num_mutation_eff(tcgaDict)
    #rawFeaturesLuad = num_mutation_eff(luadDict)
    
    rawFeatures = num_eff_per_gene(tcgaDict, False, False)
    rawFeaturesLuad = num_eff_per_gene(luadDict, False, False)
    
    # Build label vector (make label vector -1, 1)
    rawLabels = binary_labels(tcgaDict, 0)
    rawLabelsLuad = binary_labels(luadDict, 0)
    #rawLabels = class_labels(tcgaDict)
    #rawLabelsLuad = class_labels(luadDict)
    
    # Cleanup (get rid of invalid labels or features)
    rfeatureMatrix, labelVector = cleanup_inputs(rawFeatures, rawLabels)
    rfeatureMatrixLuad, labelVectorLuad = cleanup_inputs(rawFeaturesLuad, rawLabelsLuad)
    
    # Cleanup some more
    #ufeatureMatrix, ufeatureMatrixLuad = remove_useless_features( rfeatureMatrix, rfeatureMatrixLuad )
    #pfeatureMatrix, pfeatureMatrixLuad = pca_features( ufeatureMatrix, ufeatureMatrixLuad )
    #featureMatrix, featureMatrixLuad = normalize_features( pfeatureMatrix, pfeatureMatrixLuad )
    
    # Redistribute
    #rfeatureMatrix, labelVector, rfeatureMatrixLuad, labelVectorLuad = redist_data(rfeatureMatrix, labelVector, rfeatureMatrixLuad, labelVectorLuad)
    
    # From Naive Bayes, reduce feature set
    #bestIndices = np.array([201, 56, 83, 123, 42, 103, 250, 234, 217, 98, 194, 95, 113, 111, 127, 7, 50, 101, 246, 29, 30, 130, 225, 141, 78])
    #bestIndices = np.array([35, 14, 21, 32, 27, 28, 20, 11, 34, 3, 64, 57, 25, 58, 16, 24, 46, 31, 23, 1, 2, 50, 0, 26 ])
    #rfeatureMatrix = rfeatureMatrix[:,bestIndices]
    #rfeatureMatrixLuad = rfeatureMatrixLuad[:,bestIndices]
    
    # Cleanup some more
    ufeatureMatrix, ufeatureMatrixLuad = remove_useless_features( rfeatureMatrix, rfeatureMatrixLuad )
    featureMatrix, featureMatrixLuad = normalize_features( ufeatureMatrix, ufeatureMatrixLuad )
    #featureMatrix, featureMatrixLuad = pca_features( featureMatrix, featureMatrixLuad )
    
    trueLabels = one_hot_labels(labelVector)
    
    bestTestAcc = 0
    bestTrainAcc = 0
    for i in range(5000):
        
        p = np.random.permutation(len(labelVector))
        tData = featureMatrix[p,:]
        tLabels = trueLabels[p,:]
    
        devData = tData[0:100,:]
        devLabels = tLabels[0:100,:]
        trainData = tData[100:,:]
        trainLabels = tLabels[100:,:]
    
        mean = np.mean(trainData)
        std = np.std(trainData)
        trainData = (trainData - mean) / std
        devData = (devData - mean) / std
        tData = (tData - mean) / std
    
        testLabels = one_hot_labels(labelVectorLuad)
        testData = (featureMatrixLuad - mean) / std
    	
        print('Training')
        lam = 0.0
        #lam = 0.01
        params = nn_train(trainData, trainLabels, devData, devLabels, lam)
        
        ## LOAD DATA
        #params = pickle.load( open("unregularized.pickle", "rb") )
        #params = pickle.load( open("regularized.pickle", "rb") )
    
        readyForTesting = True
        if readyForTesting:
            accur = nn_test(tData, tLabels, params)
            print('Train accuracy: %f' % accur)
            accuracy = nn_test(testData, testLabels, params)
            print('Test accuracy: %f' % accuracy)
            
            if accuracy >= bestTestAcc:
                bestTestAcc = accuracy
                if accur >= bestTrainAcc:
                    bestTrainAcc = accur
                    ## SAVE DATA
                    #pickleOut = open("unregularized.pickle", "wb")
                    pickleOut = open("neural_network_simple_params.pickle", "wb")
                    pickle.dump(params, pickleOut)
                    pickleOut.close()
        #print(params)
        del params
        del devData
        del devLabels
        del trainData
        del trainLabels

if __name__ == '__main__':
    main()