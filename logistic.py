# -*- coding: utf-8 -*-
"""
Logistic Regression for Pan-Lung Data
November 16 2017
CS229 Project

Performs various logistic regression fits for data
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

"""
Logistic Regression function:
    Uses Newton's method, based off of logistic regression provided in HW#1
"""
def logistic_regression(featureMatrix, labelVector):
    # Setup
    nDat, nFeat = featureMatrix.shape
    tol = 1e-5
    maxIters = 100000
    
    # Initial guess
    thetaFit = np.zeros((nFeat+1,1))
    
    # Append one to feature matrix for intercept
    xOnes = np.ones((nDat,1))
    newFeatures = np.concatenate((xOnes, featureMatrix), axis=1)
    
    # Stopping conditions
    gradientMagnitude = 1000
    numIters = 0
    
    # Descent loop
    while( gradientMagnitude > tol ):
        
        # Useful temp vectors
        expTemp = np.exp( -np.multiply( labelVector, ( newFeatures.dot(thetaFit) )))
        hTemp = np.divide( 1, ( 1 + expTemp ) )
        hessTemp = np.multiply( labelVector, np.multiply( newFeatures, hTemp ) )
        gradTemp = np.multiply( expTemp, hessTemp )
        
        # Gradient vector
        gradVec = -1/nDat * np.sum(gradTemp, axis=0).transpose()
        gradVec.shape = (nFeat+1, 1)
        
        # Hessian Matrix
        #hessMat = np.zeros( (nFeat+1, nFeat+1) )
        #for i in range(nFeat+1):
        #    for j in range(nFeat+1):
        #        hessMat[i,j] = 1/nDat * np.sum( np.multiply( gradTemp[:,i], hessTemp[:,j] ) )
        
        # Update
        #print(hessMat)
        #print(gradVec)
        #thetaFit = thetaFit - np.linalg.inv(hessMat).dot( gradVec )
        thetaFit = thetaFit - 0.01 * gradVec
        gradientMagnitude = np.linalg.norm(gradVec)
        numIters += 1
        
        if (numIters % 100 == 0):
            print("\nIteration: ")
            print(numIters)
            print("Step: ")
            print(gradientMagnitude)
        
        if numIters > maxIters:
            break
    
    return thetaFit

def assess_accuracy( thetaFit, featureMatrix, labelVector ):
    
    # Setup
    nDat, nFeat = featureMatrix.shape
    
    # Append one to feature matrix for intercept
    xOnes = np.ones((nDat,1))
    newFeatures = np.concatenate((xOnes, featureMatrix), axis=1)
    
    # Predictions
    expTemp = np.exp( -newFeatures.dot(thetaFit) )
    hTemp = np.divide( 1, ( 1 + expTemp ) )
    predLabels = 2*(hTemp>0.5)-1
    
    # Comparison
    diff = predLabels - labelVector
    #print(np.concatenate((predLabels, labelVector, diff), axis=1))
    correctLabels = sum( diff == 0 )
    
    accuracy = correctLabels / len( labelVector )
    
    return accuracy

def plot_regression( thetaFit, featureMatrix, labelVector, title ):
    
    # Setup
    nDat, nFeat = featureMatrix.shape
    
    # Append one to feature matrix for intercept
    xOnes = np.ones((nDat,1))
    newFeatures = np.concatenate((xOnes, featureMatrix), axis=1)
    
    xData = newFeatures.dot(thetaFit)
    
    fig = plt.figure()
    plt.hold(True)
    plt.plot(xData, labelVector, 'bo')
    
    xPoints = np.linspace(min(xData), max(xData),100)
    yPoints = np.divide(1, (1+np.exp(-xPoints)))
    
    plt.plot(xPoints, yPoints, 'r')
    plt.title(title)
    plt.xlabel("Feature Mapping")
    plt.ylabel("Label / Logistic Probability")
    
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
    
    #rawFeatures = num_mutations_per_gene(tcgaDict)
    #rawFeaturesLuad = num_mutations_per_gene(luadDict)
    
    #rawFeatures = num_genes_mutated(tcgaDict)
    #rawFeaturesLuad = num_genes_mutated(luadDict)
    
    #rawFeatures = num_mutation_type(tcgaDict)
    #rawFeaturesLuad = num_mutation_type(luadDict)
    
    #rawFeatures = num_mutation_eff(tcgaDict)
    #rawFeaturesLuad = num_mutation_eff(luadDict)
    
    #rawFeatures = num_SNP(tcgaDict)
    #rawFeaturesLuad = num_SNP(luadDict)
    
    rawFeatures = num_eff_per_gene(tcgaDict, False, True)
    rawFeaturesLuad = num_eff_per_gene(luadDict, False, True)
    
    ## Special feature matrix of all features combined
    #feat1 = num_mutation_type(tcgaDict)
    #feat1Luad = num_mutation_type(luadDict)
    #feat2 = num_mutations_per_gene(tcgaDict)
    #feat2Luad = num_mutations_per_gene(luadDict)
    #feat3 = num_genes_mutated(tcgaDict)
    #feat3Luad = num_genes_mutated(luadDict)
    #feat4 = num_mutation_eff(tcgaDict)
    #feat4Luad = num_mutation_eff(luadDict)
    #feat5 = num_SNP(tcgaDict)
    #feat5Luad = num_SNP(luadDict)
    #rawFeatures = np.concatenate((feat1, feat2, feat3, feat4, feat5), axis=1)
    #rawFeaturesLuad = np.concatenate((feat1Luad, feat2Luad, feat3Luad, feat4Luad, feat5Luad), axis=1)
    
    # Build label vector (make label vector -1, 1)
    rawLabels = binary_labels(tcgaDict)
    rawLabelsLuad = binary_labels(luadDict)
    
    # Cleanup (get rid of invalid labels or features)
    featureMatrix, labelVector = cleanup_inputs(rawFeatures, rawLabels)
    featureMatrixLuad, labelVectorLuad = cleanup_inputs(rawFeaturesLuad, rawLabelsLuad)
    
    # Run logistic regression
    thetaFit = logistic_regression( featureMatrix, labelVector )
    print("Logistic Regression: ")
    print( thetaFit )
    
    # Training Accuracy
    trainAcc = assess_accuracy( thetaFit, featureMatrix, labelVector )
    print("Training Accuracy")
    print(trainAcc)
    plot_regression(thetaFit, featureMatrix, labelVector, "TCGA Training Set")
    
    # Test Accuracy
    testAcc = assess_accuracy( thetaFit, featureMatrixLuad, labelVectorLuad )
    print("Testing Accuracy")
    print(testAcc)
    plot_regression(thetaFit, featureMatrixLuad, labelVectorLuad, "LUAD Testing Set")
    
    return 0

if __name__ == '__main__':
    main()