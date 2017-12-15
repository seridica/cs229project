# -*- coding: utf-8 -*-
"""
Analysis for Neural Network for Pan-Lung Data
November 30 2017
CS229 Project

From saved parameters for complex neural network, performs some analysis.
Namely plots AUC

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
from nn_complex import *
#from nn_panlung import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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
    
    rawFeatures = num_eff_per_gene(tcgaDict, False, True)
    rawFeaturesLuad = num_eff_per_gene(luadDict, False, True)
    
    # Build label vector (make label vector -1, 1)
    rawLabels = binary_labels(tcgaDict, 0)
    rawLabelsLuad = binary_labels(luadDict, 0)
    #rawLabels = class_labels(tcgaDict)
    #rawLabelsLuad = class_labels(luadDict)
    
    # Cleanup (get rid of invalid labels or features)
    featureMatrix, labelVector = cleanup_inputs(rawFeatures, rawLabels)
    featureMatrixLuad, labelVectorLuad = cleanup_inputs(rawFeaturesLuad, rawLabelsLuad)
    
    # Cleanup some more
    #ufeatureMatrix, ufeatureMatrixLuad = remove_useless_features( rfeatureMatrix, rfeatureMatrixLuad )
    #pfeatureMatrix, pfeatureMatrixLuad = pca_features( ufeatureMatrix, ufeatureMatrixLuad )
    #featureMatrix, featureMatrixLuad = normalize_features( pfeatureMatrix, pfeatureMatrixLuad )
    
    # Redistribute
    #featureMatrix, labelVector, featureMatrixLuad, labelVectorLuad = redist_data(featureMatrix, labelVector, featureMatrixLuad, labelVectorLuad)
    
    # From Naive Bayes, reduce feature set
    #bestIndices = np.array([201, 56, 83, 123, 42, 103, 250, 234, 217, 98, 194, 95, 113, 111, 127, 7, 50, 101, 246, 29, 30, 130, 225, 141, 78])
    #bI = np.array([35, 14, 21, 32, 27, 28, 20, 11, 34, 3, 64, 57, 25, 58, 16, 24, 46, 31, 23, 1, 2, 50, 0, 26 ])
#    bI = np.array([35, 14, 21, 32, 27, 28, 20, 11, 34, 3])
#    bestIndices = np.zeros(len(bI)*4)
#    for i in range(len(bI)):
#        bestIndices[(i*4)] = bI[i]*4
#        bestIndices[(i*4)+1] = bI[i]*4+1
#        bestIndices[(i*4)+2] = bI[i]*4+2
#        bestIndices[(i*4)+3] = bI[i]*4+3
#    bestIndices = bestIndices.astype(int)
#    featureMatrix = featureMatrix[:,bestIndices]
#    featureMatrixLuad = featureMatrixLuad[:,bestIndices]
    
    geneMap = np.zeros((featureMatrix.shape[1], int(featureMatrix.shape[1]/4)))
    for i in range(int( featureMatrix.shape[1] / 4) ):
        geneMap[(i*4),i] = 1
        geneMap[(i*4+1),i] = 1
        geneMap[(i*4+2),i] = 1
        geneMap[(i*4+3),i] = 1
    
    # Cleanup some more
    #ufeatureMatrix, ufeatureMatrixLuad = remove_useless_features( featureMatrix, featureMatrixLuad )
    #featureMatrix, featureMatrixLuad = normalize_features( ufeatureMatrix, ufeatureMatrixLuad )
    #featureMatrix, featureMatrixLuad = pca_features( featureMatrix, featureMatrixLuad )
    
    mean = np.mean(featureMatrix)
    std = np.std(featureMatrix)
    featureMatrix = (featureMatrix - mean) / std
    featureMatrixLuad = (featureMatrixLuad - mean) / std
    
    testLabels = one_hot_labels(labelVectorLuad)
    trainLabels = one_hot_labels(labelVector)
    params = pickle.load(open("neural_network_params.pickle", "rb"))
    #params = pickle.load(open("neural_network_simple_params.pickle", "rb"))
    
    h, output, cost = forward_prop(featureMatrixLuad, testLabels, params)
    
    precision, recall, _ = precision_recall_curve(labelVectorLuad, output[:,1])

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    testAcc = nn_test(featureMatrixLuad, testLabels, params)
    trainAcc = nn_test(featureMatrix, trainLabels, params)
    
    average_precision = average_precision_score(labelVectorLuad, output[:,1])

    # ROC
    fpr, tpr, _ = roc_curve(labelVectorLuad, output[:,1])
    area_roc = roc_auc_score(labelVectorLuad, output[:,1])

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0,1], [0,1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.show()

    print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
    print('ROC area')
    print(area_roc)
    print('Test Accuracy:')
    print(testAcc)
    print('Train Accuracy:')
    print(trainAcc)
    
    trainData = pickle.load(open("nn_trainfeat.pickle", "rb"))
    trainLabels = pickle.load(open("nn_trainlab.pickle", "rb"))
    devData = pickle.load(open("nn_devfeat.pickle", "rb"))
    devLabels = pickle.load(open("nn_devlab.pickle", "rb"))
    
    print('Train Subset Accuracy')
    print( nn_test(trainData, trainLabels, params) )
    print('Dev Accuracy')
    print( nn_test(devData, devLabels, params))
    
    
    # Plotting for presentation
    W1 = params['W1']
    b1 = params['b1']
    plt.figure(figsize=(12,12))
    plt.imshow(W1, cmap='bwr', interpolation='none')
    plt.colorbar()
    frame1 = plt.gca()
    #frame1.axes.get_xaxis().set_visible(False)
    #frame1.axes.get_yaxis().set_visible(False)
    plt.show()
    print(W1[84:88,21])
    print(W1[8:12,2])
    print(b1[21])
    print(b1[2])
    
    W2 = params['W2']
    b2 = params['b2']
    plt.figure(figsize=(12,12))
    plt.imshow(W2, cmap='bwr', interpolation='none')
    plt.colorbar()
    frame1 = plt.gca()
    #frame1.axes.get_xaxis().set_visible(False)
    #frame1.axes.get_yaxis().set_visible(False)
    plt.show()
    print(W2[21,6])
    print(W2[2,6])
    print(b2[6])
    
    W3 = params['W3']
    plt.figure(figsize=(12,12))
    plt.imshow(W3, cmap='bwr', interpolation='none')
    plt.colorbar()
    frame1 = plt.gca()
    #frame1.axes.get_xaxis().set_visible(False)
    #frame1.axes.get_yaxis().set_visible(False)
    plt.show()
    print(W3[6,0])
    
if __name__ == '__main__':
    main()