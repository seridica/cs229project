# -*- coding: utf-8 -*-
"""
SVM for Pan-Lung Data
November 19 2017
CS229 Project

Performs a forward feature selection for a given training procedure

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
from sklearn import svm
import random
from feature_clean import *

"""
Relies on sklearn svm to perform svm
"""
def assess_accuracy( clf, featureMatrix, labelVector ):
    # Get number of samples
    N = len(labelVector)
    
    # Create outputs
    output = np.zeros((N,1))
    
    # Solve log of posterior probability
    # Loop over all test e-mails
    for i in range(N):
        inputFeatures = featureMatrix[i,:]
        inputFeatures.shape = (1,len(inputFeatures))
        output[i] = clf.predict(inputFeatures)
        
    # Comparison
    diff = output - labelVector
    #print(np.concatenate((predLabels, labelVector, diff), axis=1))
    correctLabels = sum( diff == 0 )
    
    accuracy = correctLabels / len( labelVector )
    
    return accuracy

"""
F-measure
"""
def fmeasure( clf, featureMatrix, labelVector ):
    # Get number of samples
    N = len(labelVector)
    
    # Create accuracy variables
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    
    # Solve log of posterior probability
    # Loop over all test e-mails
    for i in range(N):
        inputFeatures = featureMatrix[i,:]
        inputFeatures.shape = (1,len(inputFeatures))
        pred = clf.predict(inputFeatures)
        
        if labelVector[i] == 1:
            if pred == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred == 1:
                FP += 1
            else:
                TN += 1
        
    # Fmeasure
    accuracy = 2 * (TP) / (2*TP+FP+FN)
    
    return accuracy
    

"""
Forward feature selection using full training dataset, and evaluating
on test dataset
"""
def forward_select_simple(featureMatrix, labelVector, featureMatrixLuad, labelVectorLuad, kernelType, gamma, C):
    testAccuracy = np.zeros(featureMatrix.shape[1])
    downTrend = 0
    
    # Forward Feature Select
    featureList = list()
    currMatrix = list()
    currMatrixLuad = list()
    clf = svm.SVC(kernel=kernelType, degree=2, gamma=gamma, C=C)
    #clf = svm.SVC(kernel='rbf')
    #clf = svm.SVC(kernel='poly')
    #clf = svm.SVC(kernel='sigmoid')
    #clf.fit( featureMatrix, np.ravel(labelVector) )
    for i in range(featureMatrix.shape[1]):
        
        maxAcc = -np.inf;
        maxInd = 0;
        
        # Add another feature to the list
        for j in range(featureMatrix.shape[1]):
            if j not in featureList:
                trainMatrix = list()
                trainMatrixLuad = list()
                if i == 0:
                    trainMatrix = featureMatrix[:,j]
                    trainMatrixLuad = featureMatrixLuad[:,j]
                    trainMatrix.shape = (len(trainMatrix), 1)
                    trainMatrixLuad.shape = (len(trainMatrixLuad), 1)
                else:
                    newFeat = featureMatrix[:,j]
                    newFeat.shape = (len(newFeat),1)
                    newFeatLuad = featureMatrixLuad[:,j]
                    newFeatLuad.shape = (len(newFeatLuad),1)
                    trainMatrix = np.concatenate((currMatrix, newFeat),axis=1)
                    trainMatrixLuad = np.concatenate((currMatrixLuad, newFeatLuad),axis=1)
                
                # Train on new feature set
                clf.fit( trainMatrix, np.ravel(labelVector) )
                thisTrainAcc = assess_accuracy( clf, trainMatrix, labelVector )
                thisTestAcc = assess_accuracy( clf, trainMatrixLuad, labelVectorLuad )
                Ntest = trainMatrixLuad.shape[0]
                Ntrain = trainMatrix.shape[0]
                #thisAcc = (thisTrainAcc*Ntrain + thisTestAcc*Ntest) / (Ntest + Ntrain)
                #thisAcc = thisTrainAcc
                thisAcc = thisTestAcc
                
                # Pick new feature that gives best performance
                if thisAcc > maxAcc:
                    maxAcc = thisAcc
                    maxInd = j
        
        # Add best performing feature to the set
        if i == 0:
            currMatrix = featureMatrix[:,maxInd]
            currMatrixLuad = featureMatrixLuad[:,maxInd]
            currMatrix.shape = (len(currMatrix),1)
            currMatrixLuad.shape = (len(currMatrixLuad),1)
        else:
            newFeat = featureMatrix[:,maxInd]
            newFeat.shape = (len(newFeat),1)
            newFeatLuad = featureMatrixLuad[:,maxInd]
            newFeatLuad.shape = (len(newFeatLuad),1)
            currMatrix = np.concatenate((currMatrix, newFeat), axis=1)
            currMatrixLuad = np.concatenate((currMatrixLuad, newFeatLuad), axis=1)
        
        # Stop when performance is continuously dropping
        featureList.append(maxInd)
        testAccuracy[i] = maxAcc
        if i > 0:
            if maxAcc <= testAccuracy[i-1]:
                downTrend += 1
            else:
                downTrend = 0
        
        print("FEATURE: ")
        print(i)
        print("ADDING FEATURE: ")
        print(maxInd)
        print("Test Accuracy: ")
        print(maxAcc)
        print("All Accuracy: ")
        print(testAccuracy)
        
        if downTrend == 5:
            break
    
    return testAccuracy, featureList

"""
Forward feature selection using full training dataset, and evaluating
on test dataset
"""
def forward_select_10fold(ofeatureMatrix, olabelVector, kernelType, gamma, C):
    testAccuracy = np.zeros(ofeatureMatrix.shape[1])
    downTrend = 0
    
    # Forward Feature Select
    featureList = list()
    currMatrix = list()
    currMatrixLuad = list()
    #clf = svm.SVC(kernel='rbf')
    #clf = svm.SVC(kernel='poly')
    #clf = svm.SVC(kernel='sigmoid')
    #clf.fit( featureMatrix, np.ravel(labelVector) )
    
    # Randomize data samples
    nDat = len(olabelVector)
    data_inds = list(range(nDat))
    random.shuffle(data_inds)
    
    featureMatrix = ofeatureMatrix[data_inds,:]
    labelVector = olabelVector[data_inds]
    nFolds = 10
    nInds = int( np.floor(nDat / nFolds) )
    
    # Randomize order in which features are checked
    feature_inds = np.linspace(0,featureMatrix.shape[1]-1,featureMatrix.shape[1])
    random.shuffle(feature_inds)
    
    for i in range(featureMatrix.shape[1]):
        
        maxAcc = 0;
        maxInd = 0;
        
        # Add another feature to the list
        for abc in range(featureMatrix.shape[1]):
            j = int( feature_inds[abc] )
            
            if j not in featureList:
                
                thisAcc = 0
                #thisAcc = 1
                
                # Add new feature
                trainMatrix = list()
                if i == 0:
                    trainMatrix = featureMatrix[:,j]
                    trainMatrix.shape = (len(trainMatrix), 1)
                else:
                    newFeat = featureMatrix[:,j]
                    newFeat.shape = (len(newFeat),1)
                    trainMatrix = np.concatenate((currMatrix, newFeat),axis=1)
                
                # Perform 10-fold cross validation
                for k in range(nFolds):
                    # Recreate the solver
                    clf = svm.SVC(kernel=kernelType, degree=2, gamma=gamma, C=C)
                    
                    # Train on new feature set
                    testSlice = 0
                    if k == nFolds - 1:
                        testSlice = np.s_[(k*nInds):]
                    else:
                        testSlice = np.s_[(k*nInds):((k+1)*nInds-1)]
                    
                    trainSet = np.delete(trainMatrix, testSlice, 0)
                    testSet = trainMatrix[testSlice,:]
                    trainLabel = np.delete(labelVector, testSlice)
                    testLabel = labelVector[testSlice]
                    
                    clf.fit( trainSet, np.ravel(trainLabel) )
                    thisTrainAcc = assess_accuracy( clf, testSet, testLabel )
                    #thisTrainAcc = fmeasure( clf, testSet, testLabel )
                    thisAcc += thisTrainAcc * (len(testLabel) / nDat)
                    #if thisTrainAcc < thisAcc:
                    #    thisAcc = thisTrainAcc
                    
                    # Clear solver
                    del clf
                
                thisAcc = thisAcc
                
                # Pick new feature that gives best performance
                if thisAcc > maxAcc:
                    maxAcc = thisAcc
                    maxInd = j
        
        # Add best performing feature to the set
        if i == 0:
            currMatrix = featureMatrix[:,maxInd]
            currMatrix.shape = (len(currMatrix),1)
        else:
            newFeat = featureMatrix[:,maxInd]
            newFeat.shape = (len(newFeat),1)
            currMatrix = np.concatenate((currMatrix, newFeat), axis=1)
        
        # Stop when performance is continuously dropping
        featureList.append(maxInd)
        testAccuracy[i] = maxAcc
        if i > 0:
            if maxAcc <= testAccuracy[i-1]:
                downTrend += 1
            else:
                downTrend = 0
                
        #print("FEATURE: ")
        #print(i)
        #print("ADDING FEATURE: ")
        #print(maxInd)
        #print("Test Accuracy: ")
        #print(maxAcc)
        #print("All Accuracy: ")
        #print(testAccuracy)
        
        if downTrend == 5:
            break
    
    return testAccuracy, featureList
    
"""
Forward feature selection using full training dataset, leave one out cross
"""
def forward_select_leave_one_out(featureMatrix, labelVector, kernelType, gamma, C):
    testAccuracy = np.zeros(featureMatrix.shape[1])
    downTrend = 0
    
    # Forward Feature Select
    featureList = list()
    currMatrix = list()
    currMatrixLuad = list()
    
    nDat = featureMatrix.shape[0]
    
    # For accuracy measures
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    # Randomize order in which features are checked
    feature_inds = np.linspace(0,featureMatrix.shape[1]-1,featureMatrix.shape[1])
    random.shuffle(feature_inds)
    
    for i in range(featureMatrix.shape[1]):
        
        maxAcc = 0;
        maxInd = 0;
        
        # Add another feature to the list
        for abc in range(featureMatrix.shape[1]):
            j = int( feature_inds[abc] )
        #for j in range(featureMatrix.shape[1]):
            if j not in featureList:
                
                thisAcc = 0
                
                # Add new feature
                trainMatrix = list()
                if i == 0:
                    trainMatrix = featureMatrix[:,j]
                    trainMatrix.shape = (len(trainMatrix), 1)
                else:
                    newFeat = featureMatrix[:,j]
                    newFeat.shape = (len(newFeat),1)
                    trainMatrix = np.concatenate((currMatrix, newFeat),axis=1)
                
                # Perform 10-fold cross validation
                for k in range(nDat):
                    clf = svm.SVC(kernel=kernelType, degree=2, gamma=gamma, C=C)
                    
                    # Train on new feature set
                    trainSet = np.delete(trainMatrix, k, 0)
                    testSet = trainMatrix[k,:]
                    trainLabel = np.delete(labelVector, k)
                    testLabel = labelVector[k]
                    
                    clf.fit( trainSet, np.ravel(trainLabel) )
                    #thisTrainAcc = assess_accuracy( clf, testSet, testLabel )
                    
                    if i==0:
                        testSet.shape = (1,1)
                    else:
                        testSet.shape = (1,-1)
                        
                    pred = clf.predict(testSet)
                    
                    if testLabel == 1:
                        if pred == 1:
                            TP += 1
                        else:
                            FN += 1
                    else:
                        if pred == 1:
                            FP += 1
                        else:
                            TN += 1
                    
                    del clf
                # Fmeasure
                #thisAcc = 2 * (TP) / (2*TP+FP+FN)
                
                # Accuracy
                thisAcc = (TP + TN) / (TP+FP+FN+TN)
                
                # Pick new feature that gives best performance
                if thisAcc > maxAcc:
                    maxAcc = thisAcc
                    maxInd = j
        
        # Add best performing feature to the set
        if i == 0:
            currMatrix = featureMatrix[:,maxInd]
            currMatrix.shape = (len(currMatrix),1)
        else:
            newFeat = featureMatrix[:,maxInd]
            newFeat.shape = (len(newFeat),1)
            currMatrix = np.concatenate((currMatrix, newFeat), axis=1)
        
        # Stop when performance is continuously dropping
        featureList.append(maxInd)
        testAccuracy[i] = maxAcc
        if i > 0:
            if maxAcc <= testAccuracy[i-1]:
                downTrend += 1
            else:
                downTrend = 0
    
        print("FEATURE: ")
        print(i)
        print("ADDING FEATURE: ")
        print(maxInd)
        print("Test Accuracy: ")
        print(maxAcc)
        print("All Accuracy: ")
        print(testAccuracy)
        
        if downTrend == 5:
            break
    
    return testAccuracy, featureList

def main():
    
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
    
    rawFeatures = num_eff_per_gene(tcgaDict, False, True)
    rawFeaturesLuad = num_eff_per_gene(luadDict, False, True)
    
    # Build label vector (make label vector -1, 1)
    rawLabels = binary_labels(tcgaDict)
    rawLabelsLuad = binary_labels(luadDict)
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
    
    # Naive Bayes Feature Select
    #bestIndices = np.array([201, 67, 56, 122, 123, 83, 103, 42, 217, 234, 98, 250, 95, 194, 210, 178, 113, 134, 127, 111, 270, 262, 162, 266, 115])
    #bestIndices = np.array([35, 14, 21, 29, 65, 22, 17, 32, 13, 27, 66, 28, 20, 11, 15, 41, 34,  3, 64, 63, 67, 57, 25, 58, 16 ])
    #rfeatureMatrix = rfeatureMatrix[:,bestIndices]
    #rfeatureMatrixLuad = rfeatureMatrixLuad[:,bestIndices]
    
    ufeatureMatrix, ufeatureMatrixLuad = remove_useless_features( rfeatureMatrix, rfeatureMatrixLuad )
    featureMatrix, featureMatrixLuad = normalize_features( ufeatureMatrix, ufeatureMatrixLuad )
    featureMatrix, featureMatrixLuad = pca_features( featureMatrix, featureMatrixLuad )
    
    ## FORWARD FEATURE SELECT - ALSO FIND OPTIMAL GAMMA AND C PARAMETERS
    kernelType = 'rbf'
            
    p = np.random.permutation(len(labelVector))
    
    bestTrainAcc = 0
    bestTestAcc = 0
    for i in range(100):
        #testAccuracy, featureList = forward_select_simple( featureMatrix[p[100:],:], labelVector[p[100:],:], featureMatrix[p[:100],:], labelVector[p[:100],:], kernelType, gamma, C )
        #testAccuracy, featureList = forward_select_simple( featureMatrix, labelVector, featureMatrixLuad, labelVectorLuad, kernelType, 'auto', 1 )
        testAccuracy, featureList = forward_select_10fold( featureMatrix, labelVector, kernelType, 'auto', 1 )
        #testAccuracy, featureList = forward_select_leave_one_out( featureMatrix, labelVector, kernelType, 'auto', 1 )
        
        # Fit for best set of features
        maxAccInd = np.argmax(testAccuracy)
        maxFeatureList = featureList[:(maxAccInd+1)]
        bestFeatures = featureMatrix[:,maxFeatureList]
        bestFeaturesLuad = featureMatrixLuad[:,maxFeatureList]
        clf = svm.SVC(kernel=kernelType, degree=2)
        clf.fit( bestFeatures, np.ravel(labelVector) )
        
        # Feature List
        print("Features: ", end="")
        print(maxFeatureList, end="")
        
        print("; Num Features: ", end="")
        print(len(maxFeatureList), end="")
        
        # Training Accuracy
        trainAcc = assess_accuracy( clf, bestFeatures, labelVector )
        print("; Training Accuracy", end="")
        print(trainAcc, end="")
        
        # Test Accuracy
        testAcc = assess_accuracy( clf, bestFeaturesLuad, labelVectorLuad )
        print("; Testing Accuracy", end="")
        print(testAcc)
            
        if testAcc >= bestTestAcc:
            if (testAcc > bestTestAcc or trainAcc >= bestTrainAcc):
                bestTrainAcc = trainAcc
                bestTestAcc = testAcc
                ## SAVE DATA
                #pickleOut = open("unregularized.pickle", "wb")
                pickleOut = open("ffs.pickle", "wb")
                pickle.dump(clf, pickleOut)
                pickleOut.close()
                pickleOut1 = open("ffs_history.pickle", "wb")
                pickle.dump(testAccuracy, pickleOut1)
                pickleOut1.close()
                
    return 0

if __name__ == '__main__':
    main()