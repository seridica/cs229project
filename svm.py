# -*- coding: utf-8 -*-
"""
SVM for Pan-Lung Data
November 16 2017
CS229 Project

Performs SVM for data. Uses sci-kit learn toolbox for svm
Features and label vectors are pulled from elsewhere

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
    
def main():
    
    ## File paths etc.
    dataPath = 'D:\Stanford\CS229\Project\Data'
    
    # Load data
    tcgaDict = pickle.load(open(dataPath + "\\tcgaData.pickle", "rb"))
    luadDict = pickle.load(open(dataPath + "\luadData.pickle", "rb"))
    
    """
    Build feature matrix - Change the function to use a different feature set
    """
    #rawFeatures = num_mutations_total(tcgaDict)
    #rawFeaturesLuad = num_mutations_total(luadDict)
    
    #rawFeatures = num_genes_mutated(tcgaDict)
    #rawFeaturesLuad = num_genes_mutated(luadDict)
    
    #rawFeatures = num_mutation_type(tcgaDict)
    #rawFeaturesLuad = num_mutation_type(luadDict)
    
    #rawFeatures = num_SNP(tcgaDict)
    #rawFeaturesLuad = num_SNP(luadDict)
    
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
    rawLabels = binary_labels(tcgaDict)
    rawLabelsLuad = binary_labels(luadDict)
    #rawLabels = class_labels(tcgaDict)
    #rawLabelsLuad = class_labels(luadDict)
    
    # Cleanup (get rid of invalid labels or features)
    rfeatureMatrix, labelVector = cleanup_inputs(rawFeatures, rawLabels)
    rfeatureMatrixLuad, labelVectorLuad = cleanup_inputs(rawFeaturesLuad, rawLabelsLuad)
    
    # Naive Bayes Feature Select
    #bestIndices = np.array([201, 67, 56, 122, 123, 83, 103, 42, 217, 234, 98, 250, 95, 194, 210, 178, 113, 134, 127, 111, 270, 262, 162, 266, 115])
    #bestIndices = np.array([35, 14, 21, 29, 65, 22, 17, 32, 13, 27, 66, 28, 20, 11, 15, 41, 34,  3, 64, 63, 67, 57, 25, 58, 16 ])
    #rfeatureMatrix = rfeatureMatrix[:,bestIndices]
    #rfeatureMatrixLuad = rfeatureMatrixLuad[:,bestIndices]
    
    # More refining
    print(rfeatureMatrix.shape)
    ufeatureMatrix, ufeatureMatrixLuad = remove_useless_features( rfeatureMatrix, rfeatureMatrixLuad )
    featureMatrix, featureMatrixLuad = normalize_features( ufeatureMatrix, ufeatureMatrixLuad )
    featureMatrix, featureMatrixLuad = pca_features( featureMatrix, featureMatrixLuad )
    
    # Run logistic regression
    #clf = svm.SVC(kernel='linear')
    clf = svm.SVC(kernel='rbf')
    #clf = svm.SVC(kernel='poly')
    #clf = svm.SVC(kernel='sigmoid')
    clf.fit( featureMatrix, np.ravel(labelVector) )
    
    # Training Accuracy
    trainAcc = assess_accuracy( clf, featureMatrix, labelVector )
    print("Training Accuracy")
    print(trainAcc)
    
    # Test Accuracy
    testAcc = assess_accuracy( clf, featureMatrixLuad, labelVectorLuad )
    print("Testing Accuracy")
    print(testAcc)
    
    return 0

if __name__ == '__main__':
    main()