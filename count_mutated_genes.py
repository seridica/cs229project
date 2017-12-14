# -*- coding: utf-8 -*-
"""
Processing Code for Pan-Lung Data
November 18 2017
CS229 Project

Counts number of genes mutated total

@author: Calvin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from random import *
import io
import sys
import pickle

def main():
    ## File paths etc.
    dataPath = 'D:\Stanford\CS229\Project\Data'
    mutData = dataPath + '\Pan-Lung\data_mutations_extended.txt'
    
    # Load Cancer Cases
    typeData = dataPath + '\Pan-Lung\data_clinical.txt'
    dataCancerType = np.genfromtxt(typeData, dtype=None, delimiter='\t', usecols=(0,6,20), skip_header=6)
    cancerList = []
    for i in range(dataCancerType.shape[0]):
        cancerList.append(dataCancerType[i,0].decode("utf-8"))
    
    ## Genes
    genes = {}
    
    ## Load relevant mutation information
    dataMutations = np.genfromtxt(mutData, dtype=None, delimiter='\t', usecols=(0,15), skip_header=2)
    
    ## Fill in mutation information
    for i in range(dataMutations.shape[0]):
        
        ## Only count genes from TCGA as it is the training set
        if 'tcga' in dataMutations[i,1].decode("utf-8").lower():
            
            ## Only count adenocarcinomas
            sID = cancerList.index(dataMutations[i,1].decode("utf-8"))
            if 'adenocarcinoma' in dataCancerType[sID,2].decode("utf-8").lower():
            
                thisMutation = dataMutations[i,0].decode("utf-8").upper()
                
                currGeneKeys = list( genes.keys() )
                
                if thisMutation in currGeneKeys:
                    genes[thisMutation] = genes[thisMutation] + 1
                else:
                    genes[thisMutation] = 1
                    
        if (i%10000) == 0:
            print("\nMutation: ")
            print(i)
            
    print('\nGene Counts: ')
    print(len(genes))
        
    # Get genes that appear in >10% of cases
    tempGenes = sorted(genes, key=genes.get, reverse=True)
    i = 0
    #tenPercent = dataCancerType.shape[0] * 0.25
    tenPercent = 501 * 0.25 # Only count TCGA
    while genes[tempGenes[i]] > tenPercent:
        print("\nGene: ")
        print(tempGenes[i])
        print(genes[tempGenes[i]])
        i += 1
    
    print("\nTotal Genes: ")
    print(i)
    dataPath = 'D:\Stanford\CS229\Project\Data'
    pickle1 = open(dataPath + "\geneList.pickle", "wb")
    pickle.dump(tempGenes[:i], pickle1)
    pickle1.close()
    return

if __name__ == '__main__':
    main()