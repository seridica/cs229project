# -*- coding: utf-8 -*-
"""
Count features
November 18 2017
CS229 Project

This function processes the data for a number of features related to
count. E.g.

1) Number of mutations total
2) Number of mutations per gene
3) Number of mutations per type
4) Number of mutations per class
5) Number of SNP mutations
6) Number of genes mutated

@author: Calvin
"""
import numpy as np
import pickle

"""
Counts total number of mutations in genes of interest for each case.
Number of features = 1
"""
def num_mutations_total(dataIn):
    
    # Generate Feature Matrix
    nDat = len( dataIn )
    featureMatrix = np.zeros((nDat, 1))
    dictKeys = list(dataIn.keys())
    
    # Cycle through cases
    for i in range(nDat):
        totalMutations = 0
        genes = list(dataIn[dictKeys[i]]['genes'].keys())
        
        # Cycle through genes
        for gene in genes:
            totalMutations += len( dataIn[dictKeys[i]]['genes'][gene] )
        
        # Count mutations in each gene and add to total
        featureMatrix[i,0] = totalMutations
    
    return featureMatrix

"""
Counts total number of mutations in each gene of interest for each case.
Number of features = number of genes of interest
"""
def num_mutations_per_gene(dataIn):
    
    # Generate Feature Matrix
    nDat = len( dataIn )
    dictKeys = list(dataIn.keys())
    genes = list(dataIn[dictKeys[0]]['genes'].keys())
    nGenes = len(genes)
    
    featureMatrix = np.zeros((nDat, nGenes))
    
    # Cycle through cases
    for i in range(nDat):
        
        # Cycle through genes
        for j in range(nGenes):
            
            # Add gene mutation count to feature matrix
            featureMatrix[i,j] = len( dataIn[dictKeys[i]]['genes'][genes[j]] )
    
    return featureMatrix

"""
Counts total number of mutations of a certain type (SNP, DNP, etc.) per case
in genes of interest
features = number of mutation types
"""
def num_mutation_type(dataIn):
    
    # Get list of mutation types, note this includes the "other" classification
    dataPath = 'D:\Stanford\CS229\Project\Data'
    mutTypes = pickle.load(open(dataPath + "\mutationTypes.pickle", "rb"))
    types = list(mutTypes.keys())
    
    # Generate Feature Matrix
    nDat = len(dataIn)
    dictKeys = list(dataIn.keys())
    featureMatrix = np.zeros((nDat, len(types)-1))
    
    for i in range(nDat):
        
        genes = list(dataIn[dictKeys[i]]['genes'].keys())
        
        for gene in genes:
            mutList = dataIn[dictKeys[i]]['genes'][gene]
            
            for k in range(len(mutList)):
                thisMutation = mutList[k]
                if (thisMutation['type'] in types):
                    ind = types.index(thisMutation['type'])
                    if ind == 4:
                        featureMatrix[i,0] += np.inf
                    else:
                        featureMatrix[i,ind] += 1
                else:
                    featureMatrix[i,0] = np.inf
    
    return featureMatrix

"""
Counts total number of mutations of a certain effect (missense, nonsense) per
case in genes of interest
features = number of effect classes
"""
def num_mutation_eff(dataIn):
    
    # Get list of mutation types, note this includes the "other" classification
    dataPath = 'D:\Stanford\CS229\Project\Data'
    mutEffects = pickle.load(open(dataPath + "\mutationEffects.pickle", "rb"))
    effects = list(mutEffects.keys())
    
    # Generate Feature Matrix
    nDat = len(dataIn)
    dictKeys = list(dataIn.keys())
    featureMatrix = np.zeros((nDat, len(effects)-1))
    
    for i in range(nDat):
        
        genes = list(dataIn[dictKeys[i]]['genes'].keys())
        
        for gene in genes:
            mutList = dataIn[dictKeys[i]]['genes'][gene]
            
            for k in range(len(mutList)):
                thisMutation = mutList[k]
                if (thisMutation['class'] in effects):
                    ind = effects.index(thisMutation['class'])
                    if ind == 6:
                        featureMatrix[i,0] = np.inf
                    else:
                        featureMatrix[i,ind] += 1
                else:
                    featureMatrix[i,0] = np.inf
    
    return featureMatrix

"""
Counts total number of SNP mutation types
features = 12
"""
def num_SNP(dataIn):
    
    # Generate Feature Matrix
    nDat = len( dataIn )
    dictKeys = list(dataIn.keys())
    
    # Get list of SNP mutations
    dataPath = 'D:\Stanford\CS229\Project\Data'
    mutDict = pickle.load(open(dataPath + "\mutationList.pickle", "rb"))
    mutList = list(mutDict.keys())
    
    featureMatrix = np.zeros((nDat, len(mutList)))
    
    # Cycle through cases
    for i in range(nDat):
        genes = list(dataIn[dictKeys[i]]['genes'].keys())
        
        # Cycle through genes
        for gene in genes:
            thisMutList = dataIn[dictKeys[i]]['genes'][gene]
            
            # If mutation is an SNP, include it in count
            for k in range(len(thisMutList)):
                thisMutation = thisMutList[k]
                if (thisMutation['mutation'] in mutList):
                    ind = mutList.index(thisMutation['mutation'])
                    featureMatrix[i,ind] += 1
    
    return featureMatrix

"""
Counts total number of genes that were mutated
features = 1
"""
def num_genes_mutated(dataIn):
    
    # Generate Feature Matrix
    nDat = len( dataIn )
    dictKeys = list(dataIn.keys())
    genes = list(dataIn[dictKeys[0]]['genes'].keys())
    nGenes = len(genes)
    
    featureMatrix = np.zeros((nDat, 1))
    
    # Cycle through cases
    for i in range(nDat):
        totalGenesMutated = 0
        
        # Cycle through genes
        for j in range(nGenes):
            
            # Add gene mutation count to feature matrix
            totalGenesMutated += (len( dataIn[dictKeys[i]]['genes'][genes[j]] ) > 0)
    
        featureMatrix[i] = totalGenesMutated
    
    return featureMatrix

"""
Counts total number of mutation effects in each gene.
This feature is thus a combination of num_mutation_eff and num_mutations_per_gene

To reduce feature set, can also set to ignore silent mutations, also can
set to a boolean input
"""
def num_eff_per_gene(dataIn, silent=False, yesno=False):
    
    # Generate Feature Matrix
    nDat = len( dataIn )
    dictKeys = list(dataIn.keys())
    genes = list(dataIn[dictKeys[0]]['genes'].keys())
    nGenes = len(genes)
        
    # Get list of mutation types, note this includes the "other" classification
    dataPath = 'D:\Stanford\CS229\Project\Data'
    mutEffects = pickle.load(open(dataPath + "\mutationEffects.pickle", "rb"))
    effects = list(mutEffects.keys())[:-2]
    
    # Generate Feature Matrix
    nDat = len(dataIn)
    dictKeys = list(dataIn.keys())
    
    # Include silent mutations
    if silent:
        featureMatrix = np.zeros((nDat, (len(effects))*nGenes))
    else:
        featureMatrix = np.zeros((nDat, (len(effects)-1)*nGenes))
    
    # Fill feature matrix
    for i in range(nDat):
        nGene = 0
        for gene in genes:
            mutList = dataIn[dictKeys[i]]['genes'][gene]
            
            for k in range(len(mutList)):
                thisMutation = mutList[k]
                if (thisMutation['class'] in effects):
                    ind = effects.index(thisMutation['class'])
                    
                    # Include silent mutations
                    full_ind = 0
                    if silent:
                        full_ind = nGene * (len(effects)) + ind
                    else:
                        full_ind = nGene * (len(effects)-1) + ind - 1
                        
                    # Binary Input
                    if yesno:
                        featureMatrix[i,full_ind] = 1
                    else:
                        featureMatrix[i,full_ind] += 1
            
            nGene += 1
    
    return featureMatrix

"""
This determines whether there is a mutation in a particular codon
of a gene. This feature set is special in that it also returns
the features associated with each gene.
"""
def mutation_codon(dataIn):
    
    # Generate Feature Matrix
    nDat = len( dataIn )
    dictKeys = list(dataIn.keys())
    genes = list(dataIn[dictKeys[0]]['genes'].keys())
    nGenes = len(genes)
        
    # Get list of mutation types, note this includes the "other" classification
    dataPath = 'D:\Stanford\CS229\Project\Data'
    mutEffects = pickle.load(open(dataPath + "\mutationEffects.pickle", "rb"))
    effects = list(mutEffects.keys())[:-2]
    
    # Generate Feature Matrix
    nDat = len(dataIn)
    dictKeys = list(dataIn.keys())
    
    # Include silent mutations
    if silent:
        featureMatrix = np.zeros((nDat, (len(effects))*nGenes))
    else:
        featureMatrix = np.zeros((nDat, (len(effects)-1)*nGenes))
    
    # Fill feature matrix
    for i in range(nDat):
        nGene = 0
        for gene in genes:
            mutList = dataIn[dictKeys[i]]['genes'][gene]
            
            for k in range(len(mutList)):
                thisMutation = mutList[k]
                if (thisMutation['class'] in effects):
                    ind = effects.index(thisMutation['class'])
                    
                    # Include silent mutations
                    full_ind = 0
                    if silent:
                        full_ind = nGene * (len(effects)) + ind
                    else:
                        full_ind = nGene * (len(effects)-1) + ind - 1
                        
                    # Binary Input
                    if yesno:
                        featureMatrix[i,full_ind] = 1
                    else:
                        featureMatrix[i,full_ind] += 1
            
            nGene += 1
            
    return featureMatrix, codonDict