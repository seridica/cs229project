# -*- coding: utf-8 -*-
"""
Naive Bayes for Pan-Lung Data
November 16 2017
CS229 Project

Performs Naive Bayes for data, based on homework 2 implementation
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
Naive Bayes:
    This is exactly the implementation I had in HW#2. With Laplace Smoothing
"""
def naive_bayes(featureMatrix, labelVector):
    # Get number of samples
    M, N = featureMatrix.shape
    
    # State matrix represents the counts for each tolken
    # Storing counts in state instead of the actual probabilities for
    # log-likelihood calculation later
    # Each row is a count for a word, column represents counts for spam
    # vs. not spam
    # Extra two states represent:
    # Penultimate row - total number of rows in spam vs. not spam
    # Last row - Total count spam vs. not spam
    state = np.zeros((N+2,2))
    
    # Count words
    for i in range(M):
        wordCounts = np.transpose(featureMatrix[i,:])
        totalWords = np.sum(wordCounts)
        if labelVector[i]==-1:
            state[:-2,0] += wordCounts
            state[-1,0] += 1
            state[-2,0] += totalWords
        else:
            state[:-2,1] += wordCounts
            state[-1,1] += 1
            state[-2,1] += totalWords
    
    # Laplace smoothing
    for i in range(N):
        state[i,0] += 1
        state[i,1] += 1

    # Add number of tolkens to denominator values
    state[-2,0] += N
    state[-2,1] += N
    
    return state

def assess_accuracy( state, featureMatrix, labelVector ):
    # Get number of samples
    M, N = featureMatrix.shape
    
    # Create outputs
    output = np.zeros((M,1))
    
    # Solve log of posterior probability
    # Loop over all test e-mails
    for i in range(M):
        p0 = 0
        p1 = 0
        
        # Loop over each word
        for j in range(N):
            
            # From log probability, sum log numerators scaled by word count
            p0 += featureMatrix[i,j]*np.log(state[j,0])
            p1 += featureMatrix[i,j]*np.log(state[j,1])
    
        # Subtract denominators
        totWords = np.sum(featureMatrix[i,:])
        p0 -= np.log( state[-2,0] )*totWords;
        p1 -= np.log( state[-2,1] )*totWords;
        
        # Add prior probability
        totM = state[-1,0] + state[-1,1]
        p0 += ( np.log( state[-1,0] ) - np.log( totM ) )
        p1 += ( np.log( state[-1,1] ) - np.log( totM ) )

        if p1 > p0:
            output[i]= 1
        else:
            output[i] = -1
    
    # Comparison
    diff = output - labelVector
    #print(np.concatenate((predLabels, labelVector, diff), axis=1))
    correctLabels = sum( diff == 0 )
    
    accuracy = correctLabels / len( labelVector )
    
    return accuracy
    
def best_features(state, tokenlist):
    N = len(tokenlist)
    tokenscores = np.zeros(N)
    
    # Compute the score for spam words
    for i in range(N):
        tokenscores[i] = np.log(state[i,1]) - np.log(state[-2,1]) - ( np.log(state[i,0]) - np.log(state[-2,0]) )
        
    # Sort
    indices = np.argsort(tokenscores)
    
    # Print top 5
    print("\nMost Predictive: ")
    for i in range(5):
        print(tokenlist[indices[-(1+i)]])
        
    # Print bottom 5
    print("\nLeast Predictive: ")
    for i in range(10):
        print(tokenlist[indices[i]])
        
    return np.flip(indices, 0)

def main():
    
    ## File paths etc.
    dataPath = 'D:\Stanford\CS229\Project\Data'
    
    # Load data
    tcgaDict = pickle.load(open(dataPath + "\\tcgaData.pickle", "rb"))
    luadDict = pickle.load(open(dataPath + "\luadData.pickle", "rb"))
    
    """
    Build feature matrix - Change the function to use a different feature set
    """
    rawFeatures = num_mutations_per_gene(tcgaDict)
    rawFeaturesLuad = num_mutations_per_gene(luadDict)
    genes = pickle.load(open(dataPath + "\genes.pickle", "rb"))
    tokenList = list(genes.keys())[:-1]
    
    #rawFeatures = num_mutation_type(tcgaDict)
    #rawFeaturesLuad = num_mutation_type(luadDict)
    #types = pickle.load(open(dataPath + "\mutationTypes.pickle", "rb"))
    #tokenList = list(types.keys())[:-1]
    
    #rawFeatures = num_mutation_eff(tcgaDict)
    #rawFeaturesLuad = num_mutation_eff(luadDict)
    #effs = pickle.load(open(dataPath + "\mutationEffects.pickle", "rb"))
    #tokenList = list(effs.keys())[:-1]
    
    #rawFeatures = num_SNP(tcgaDict)
    #rawFeaturesLuad = num_SNP(luadDict)
    #mutList = pickle.load(open(dataPath + "\mutationList.pickle", "rb"))
    #tokenList = list(mutList.keys())
    
    #rawFeatures = num_eff_per_gene(tcgaDict, False, False)
    #rawFeaturesLuad = num_eff_per_gene(luadDict, False, False)
    #genes = pickle.load(open(dataPath + "\genes.pickle", "rb"))
    #effs = pickle.load(open(dataPath + "\mutationEffects.pickle", "rb"))
    #geneList = list(genes.keys())[:-1]
    #effList = list(effs.keys())[1:-2]
    #tokenList = [x + y for x in geneList for y in effList]
    
    # Build label vector (make label vector -1, 1)
    rawLabels = binary_labels(tcgaDict)
    rawLabelsLuad = binary_labels(luadDict)
    
    # Cleanup (get rid of invalid labels or features)
    featureMatrix, labelVector = cleanup_inputs(rawFeatures, rawLabels)
    featureMatrixLuad, labelVectorLuad = cleanup_inputs(rawFeaturesLuad, rawLabelsLuad)
    
    # Select best (from a previous run of Naive Bayes with full set)
    #bestIndices = np.array([201, 56, 83, 123, 42, 103, 250, 234, 217, 98, 194, 95, 113, 111, 127, 7, 50, 101, 246, 29, 30, 130, 225, 141, 78])
    #print(bestIndices)
    #featureMatrix = featureMatrix[:,bestIndices]
    #featureMatrixLuad = featureMatrixLuad[:,bestIndices]
    #tokenList = tokenList[bestIndices]
    
    # Run logistic regression
    state = naive_bayes( featureMatrix, labelVector )
    print("Naive Bayes: ")
    print( state )
    
    # Training Accuracy
    trainAcc = assess_accuracy( state, featureMatrix, labelVector )
    print("Training Accuracy")
    print(trainAcc)
    
    # Test Accuracy
    testAcc = assess_accuracy( state, featureMatrixLuad, labelVectorLuad )
    print("Testing Accuracy")
    print(testAcc)
    
    # Most important features
    featureOrder = best_features(state, tokenList)
    
    print(featureOrder)
    
    return 0

if __name__ == '__main__':
    main()