# -*- coding: utf-8 -*-
"""
SVM for Pan-Lung Data
November 30 2017
CS229 Project

File provides functions for cleaning up feature set.

1) Function removes features that have all the same value (typically 0)
2) PCA on features
3) Normalize features

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
from cleanup import *
import pdb
from sklearn.decomposition import PCA
import random

"""
Function removes features that have all the same value.
"""
def remove_useless_features(trainFeatureMatrix, testFeatureMatrix):
    
    nData, nFeat = trainFeatureMatrix.shape
    nTest = testFeatureMatrix.shape[0]
    newTrainFeatureMatrix = np.array([[]])
    newTestFeatureMatrix = np.array([[]])
    newTrainFeatureMatrix.shape = (nData, 0)
    newTestFeatureMatrix.shape = (nTest, 0)
    for i in range(nFeat):
        tot_Sum = np.sum(trainFeatureMatrix[:,i])
        if not (tot_Sum % nData == 0):
            ntrainf = trainFeatureMatrix[:,i]
            ntrainf.shape = (nData, 1)
            ntestf = testFeatureMatrix[:,i]
            ntestf.shape = (nTest, 1)
            newTrainFeatureMatrix = np.concatenate( (newTrainFeatureMatrix, ntrainf), axis=1 )
            newTestFeatureMatrix = np.concatenate( (newTestFeatureMatrix, ntestf), axis=1 )
    
    return newTrainFeatureMatrix, newTestFeatureMatrix

"""
Function performs PCA on the features to identify the where all the variance
in the data lies.
"""
def pca_features(trainFeatureMatrix, testFeatureMatrix):
    pca = PCA()
    fullTrainPCA = pca.fit_transform(trainFeatureMatrix)
    fullTestPCA = pca.transform(testFeatureMatrix)
    
    expVar = pca.explained_variance_ratio_
    print(expVar)
    
    cumExp = 0
    thresh = 0.9
    for i in range(len(expVar)):
        cumExp += expVar[i]
        if cumExp > thresh:
            break;
    
    #thresh = 0.1
    #for i in range(len(expVar)):
    #    if expVar[i] < thresh:
    #        break;
    
    print("Number of components: ")
    print( i )
    print("Number of original features: ")
    print(trainFeatureMatrix.shape[1])
    
    newTrainFeatureMatrix = fullTrainPCA[:,:i]
    newTestFeatureMatrix = fullTestPCA[:,:i]
    
    # Plotting for presentation
    components = (pca.components_)
    plt.figure(figsize=(12,12))
    plt.imshow(components, cmap='bwr', interpolation='none')
    plt.colorbar()
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.show()
    
    return newTrainFeatureMatrix, newTestFeatureMatrix

"""
Function demeans and normalizes features
"""
def normalize_features(trainFeatureMatrix, testFeatureMatrix):
    nDat, nFeat = trainFeatureMatrix.shape
    
    newTrainFeatureMatrix = np.array(trainFeatureMatrix)
    newTestFeatureMatrix = np.array(testFeatureMatrix)
    
    for i in range(nFeat):
        thisFeat = trainFeatureMatrix[:,i]
        
        mFeat = np.mean(thisFeat)
        mStd = np.std(thisFeat)
        thisFeat = ( thisFeat - mFeat ) / mStd
        
        newTrainFeatureMatrix[:,i] = thisFeat
        
        testFeat = testFeatureMatrix[:,i]
        if mStd == 0:
            testFeat = testFeat - mFeat
        else:
            testFeat = ( testFeat - mFeat ) / mStd
        
        newTestFeatureMatrix[:,i] = testFeat
        
    return newTrainFeatureMatrix, newTestFeatureMatrix

"""
Function redistributes test and train data
"""
def redist_data( trainData, trainLabels, testData, testLabels ):
    
    newRatio = 5
    
    trainClassInds = {}
    testClassInds = {}
    
    nTrainDat, nFeatures = trainData.shape
    nTestDat = testData.shape[0]
    
    # Partition data in train and test
    trainKeys = []
    for i in range(nTrainDat):
        currLabel = int( trainLabels[i] )
        if currLabel in trainKeys:
            trainClassInds[currLabel].append(i)
        else:
            trainClassInds[currLabel] = [i]
            trainKeys.append(currLabel)
            
    testKeys = []
    for i in range(nTestDat):
        currLabel = int( testLabels[i] )
        if currLabel in testKeys:
            testClassInds[currLabel].append(i)
        else:
            testClassInds[currLabel] = [i]
            testKeys.append(currLabel)
    
    # Make sure there are the same number of class labels
    assert( len(testKeys) == len(trainKeys) )
    
    # Redistribute
    newTrainData = np.array([[]])
    newTrainData.shape = (0, nFeatures)
    newTrainLabels = []
    newTestData = np.array([[]])
    newTestData.shape = (0, nFeatures)
    newTestLabels = []
    for i in range(len(testKeys)):
        # For original training data
        inds = np.array(trainClassInds[testKeys[i]])
        p = np.random.permutation(len(inds))
        
        cutoff = int( np.floor( len(inds) / newRatio ) )
        newTrainData = np.concatenate( (newTrainData, trainData[inds[p[cutoff:]],:] ), axis=0 )
        newTestData = np.concatenate( (newTestData, trainData[inds[p[:cutoff]],:] ), axis=0 )
        
        newTrainLabels = np.concatenate( (newTrainLabels, trainLabels[inds[p[cutoff:]]].reshape(-1)) )
        newTestLabels = np.concatenate( (newTestLabels, trainLabels[inds[p[:cutoff]]].reshape(-1)) )
        
        # For original test data
        inds = np.array(testClassInds[testKeys[i]])
        p = np.random.permutation(len(inds))
        
        cutoff = int( np.floor( len(inds) / newRatio ) )
        newTrainData = np.concatenate( (newTrainData, testData[inds[p[cutoff:]],:] ), axis=0 )
        newTestData = np.concatenate( (newTestData, testData[inds[p[:cutoff]],:] ), axis=0 )
        
        newTrainLabels = np.concatenate( (newTrainLabels, testLabels[inds[p[cutoff:]]].reshape(-1)) )
        newTestLabels = np.concatenate( (newTestLabels, testLabels[inds[p[:cutoff]]].reshape(-1)) )
    
    print( newTrainData.shape )
    print( newTestData.shape )
    
    newTrainLabels = np.array([newTrainLabels]).T
    newTestLabels = np.array([newTestLabels]).T
    
    return newTrainData, newTrainLabels, newTestData, newTestLabels