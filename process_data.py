# -*- coding: utf-8 -*-
"""
Processing Code for Pan-Lung Data
November 12 2017
CS229 Project

Pull data from sources to build master data dictionaries. One dictionary for
each dataset (TGCA and LUAD)
Dictionary Organization:
    sample ID:
        genes:
            gene name: list of mutations
                type: mutation type
                class: mutation classification
                loc: mutation location in specific gene
                mutation: specific mutation
        grade: Cancer grade

Note, for both genes and mutations, only significantly mutated genes are
included as well as any mutations within those genes based on prior work.
(Ding 2008, TGCA 2016). Specifics below

Genes:
    TP53, KRAS, STK11, EGFR, LRP1B, NF1, ATM, APC, EPHA3, PTPRD, CDKN2A, ERBB4,
    KDR, FGFR4, NTRK1, EPHA5, PDGFRA, GNAS, LTK, INHBA, PAK3, ZMYND10, NRAS,
    SLC38A3, KEAP1, BRAF, SETD2, RBM10, MGA, MET, ARID1A, PIK3CA, SMARCA4,
    RB1, U2AF1, RIT1

Mutation types:
    SNP (single nucleotide polymorphism), DNP (double nucleotide polymorphism),
    DEL (deletion), INS (insertion)

Mutation Classes:
    Silent, Missense, Nonsense, Frame Shift, Splice

@author: Calvin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from random import *
import io
import sys
import pickle

def build_mutation(mutationLine, mutTypes, mutEff):
    
    thisType = mutationLine[2].decode("utf-8")
    thisClass = mutationLine[1].decode("utf-8")
    
    correctedType = 'Other'
    correctedClass = 'Other'
    for i in range(len(mutTypes)):
        compType = mutTypes[i]
        if compType.lower() in thisType.lower():
            correctedType = compType
            break;
    
    for i in range(len(mutEff)):
        compEff = mutEff[i]
        if compEff.lower() in thisClass.lower():
            correctedClass = compEff
            break;
        
    wld = mutationLine[3].decode("utf-8")
    ale1 = mutationLine[4].decode("utf-8")
    ale2 = mutationLine[5].decode("utf-8")
    
    loc = mutationLine[7].decode("utf-8").split('/')[0]
    
    # Get only first location index
    if len(loc.split('-')) > 1:
        loc = loc.split('-')[0]
    
    mut1 = {'type': correctedType,
            'class': correctedClass,
            'loc': loc,
            'mutation': wld+ale2}
    
    if wld != ale1:    
        mut2 = {'type': correctedType,
                'class': correctedClass,
                'loc': loc,
                'mutation': wld + ale1}
    else:
        mut2 = {}
    
    return mut1, mut2

def main():
    ## File paths etc.
    dataPath = 'D:\Stanford\CS229\Project\Data'
    typeData = dataPath + '\Pan-Lung\data_clinical.txt'
    mutData = dataPath + '\Pan-Lung\data_mutations_extended.txt'
    
    ## Load genes with >10% representation
    geneList = pickle.load(open(dataPath + "\geneList.pickle", "rb"))
    
    ## Genes
    genes = {'TP53': {'luad': 0, 'tcga': 0},
             'KRAS': {'luad': 0, 'tcga': 0},
             'STK11': {'luad': 0, 'tcga': 0},
             'EGFR': {'luad': 0, 'tcga': 0},
             'LRP1B': {'luad': 0, 'tcga': 0},
             'NF1': {'luad': 0, 'tcga': 0},
             'ATM': {'luad': 0, 'tcga': 0},
             'APC': {'luad': 0, 'tcga': 0},
             'EPHA3': {'luad': 0, 'tcga': 0},
             'PTPRD': {'luad': 0, 'tcga': 0},
             'CDKN2A': {'luad': 0, 'tcga': 0},
             'ERBB4': {'luad': 0, 'tcga': 0},
             'KDR': {'luad': 0, 'tcga': 0},
             'FGFR4': {'luad': 0, 'tcga': 0},
             'NTRK1': {'luad': 0, 'tcga': 0},
             'EPHA5': {'luad': 0, 'tcga': 0},
             'PDGFRA': {'luad': 0, 'tcga': 0},
             'GNAS': {'luad': 0, 'tcga': 0},
             'LTK': {'luad': 0, 'tcga': 0},
             'INHBA': {'luad': 0, 'tcga': 0},
             'PAK3': {'luad': 0, 'tcga': 0},
             'ZMYND10': {'luad': 0, 'tcga': 0},
             'NRAS': {'luad': 0, 'tcga': 0},
             'SLC38A3': {'luad': 0, 'tcga': 0},
             'KEAP1': {'luad': 0, 'tcga': 0},
             'BRAF': {'luad': 0, 'tcga': 0},
             'SETD2': {'luad': 0, 'tcga': 0},
             'RBM10': {'luad': 0, 'tcga': 0},
             'MGA': {'luad': 0, 'tcga': 0},
             'MET': {'luad': 0, 'tcga': 0},
             'ARID1A': {'luad': 0, 'tcga': 0},
             'PIK3CA': {'luad': 0, 'tcga': 0},
             'SMARCA4': {'luad': 0, 'tcga': 0},
             'RB1': {'luad': 0, 'tcga': 0},
             'U2AF1': {'luad': 0, 'tcga': 0},
             'RIT1': {'luad': 0, 'tcga': 0}}
    
    # Add genes with >10% representation
    geneKeys = list(genes.keys())
    for gene in geneList:
        if gene.upper() not in geneKeys:
            genes[gene.upper()] = {'luad':0, 'tcga': 0}
    genes['Other'] = {'luad': 0, 'tcga': 0}
    
    geneKeys = list(genes.keys())
    
#    genes = {'RIT1': {'luad': 0, 'tcga': 0},
#             'NTRK1': {'luad': 0, 'tcga': 0},
#             'ZMYND10': {'luad': 0, 'tcga': 0},
#             'SMARCA4': {'luad': 0, 'tcga': 0},
#             'RBM10': {'luad': 0, 'tcga': 0},
#             'MGA': {'luad': 0, 'tcga': 0},
#             'PAK3': {'luad': 0, 'tcga': 0},
#             'ERBB4': {'luad': 0, 'tcga': 0},
#             'U2AF1': {'luad': 0, 'tcga': 0},
#             'EGFR': {'luad': 0, 'tcga': 0},
#             'Other': {'luad': 0, 'tcga': 0}}
#    geneKeys = list(genes.keys())
    
    ## Cancer Classes
    cancerClass = {'IA': {'luad': 0, 'tcga': 0},
                   'IB': {'luad': 0, 'tcga': 0},
                   'IIA': {'luad': 0, 'tcga': 0},
                   'IIB': {'luad': 0, 'tcga': 0},
                   'IIIA': {'luad': 0, 'tcga': 0},
                   'IIIB': {'luad': 0, 'tcga': 0},
                   'IV': {'luad': 0, 'tcga': 0},
                   'Other': {'luad': 0, 'tcga': 0}}
    
    ## Mutation Types
    mutTypes = {'SNP': {'luad': 0, 'tcga': 0},
                'DNP': {'luad': 0, 'tcga': 0},
                'INS': {'luad': 0, 'tcga': 0},
                'DEL': {'luad': 0, 'tcga': 0},
                'Other': {'luad': 0, 'tcga': 0}}
    
    ## Mutation Effect
    mutEff = {'Silent': {'luad': 0, 'tcga': 0},
              'Missense': {'luad': 0, 'tcga': 0},
              'Nonsense': {'luad': 0, 'tcga': 0},
              'Shift': {'luad': 0, 'tcga': 0},
              'Splice': {'luad': 0, 'tcga': 0},
              'Intron': {'luad': 0, 'tcga': 0},
              'Other': {'luad': 0, 'tcga': 0}}
    
    ## Start a new list for mutations
    mutList = {'AC': {'luad': 0, 'tcga': 0},
               'AG': {'luad': 0, 'tcga': 0},
               'AT': {'luad': 0, 'tcga': 0},
               'CA': {'luad': 0, 'tcga': 0},
               'CG': {'luad': 0, 'tcga': 0},
               'CT': {'luad': 0, 'tcga': 0},
               'GA': {'luad': 0, 'tcga': 0},
               'GC': {'luad': 0, 'tcga': 0},
               'GT': {'luad': 0, 'tcga': 0},
               'TA': {'luad': 0, 'tcga': 0},
               'TC': {'luad': 0, 'tcga': 0},
               'TG': {'luad': 0, 'tcga': 0}}
    
    ## Load Data Cancer Type Data - Only type data
    dataCancerType = np.genfromtxt(typeData, dtype=None, delimiter='\t', usecols=(0,6,20), skip_header=6)
    
    luadDict = {}
    tcgaDict = {}
    
    numLuadCases = 0
    numTcgaCases = 0
    numExtraCases = 0
    
    ## Fill cancer grade and prepare dictionary for loading mutations
    for i in range(dataCancerType.shape[0]):
        
        # Check if adenocarcinoma
        thisCancer = dataCancerType[i,2].decode("utf-8")
        if ("adenocarcinoma" in thisCancer.lower()):
            
            cancerType = dataCancerType[i,1].decode("utf-8")
            newDict = {'genes': {}, 'grade': cancerType }
            
            for j in range(len(geneKeys)-1):
                newDict['genes'][geneKeys[j]] = []
            
            sampleID = dataCancerType[i,0].decode("utf-8")
            if 'luad' in sampleID.lower():
                # Check for overlapping cases
                if sampleID in luadDict:
                    print('Already Here: LUAD ' + sampleID)
                luadDict[sampleID] = newDict
                numLuadCases += 1
                if cancerType in list(cancerClass.keys()):
                    cancerClass[cancerType]['luad'] += 1
                else:
                    cancerClass['Other']['luad'] += 1
            else:
                if 'tcga' in sampleID.lower():
                    # Check for overlapping cases
                    if sampleID in tcgaDict:
                        print('Already Here: TCGA ' + sampleID)
                    tcgaDict[sampleID] = newDict
                    numTcgaCases += 1
                    if cancerType in list(cancerClass.keys()):
                        cancerClass[cancerType]['tcga'] += 1
                    else:
                        cancerClass['Other']['tcga'] += 1
                else:
                    print(sampleID)
                    numExtraCases += 1
    
    ## Load relevant mutation information
    dataMutations = np.genfromtxt(mutData, dtype=None, delimiter='\t', usecols=(0,8,9,10,11,12,15,58), skip_header=2)
    luadIDs = list(luadDict.keys())
    tcgaIDs = list(tcgaDict.keys())
    
    ## Fill in mutation information
    for i in range(dataMutations.shape[0]):
        sampleID = dataMutations[i,6].decode("utf-8")
        thisMutation = dataMutations[i,0].decode("utf-8") 
        if 'luad' in sampleID.lower():
            if sampleID in luadIDs:
                if thisMutation.upper() in geneKeys:
                    
                    mutInfo1, mutInfo2 = build_mutation(dataMutations[i,:], list(mutTypes.keys()), list(mutEff.keys()))
                    
                    luadDict[sampleID]['genes'][thisMutation.upper()].append(mutInfo1)
                    mutTypes[mutInfo1['type']]['luad'] += 1
                    mutEff[mutInfo1['class']]['luad'] += 1
                    if mutInfo1['mutation'] in list(mutList.keys()):
                        mutList[mutInfo1['mutation']]['luad'] += 1
                    
                    if mutInfo2:
                        print(mutInfo2)
                        luadDict[sampleID]['genes'][thisMutation.upper()].append(mutInfo2)
                        mutTypes[mutInfo2['type']]['luad'] += 1
                        mutEff[mutInfo2['class']]['luad'] += 1
                        if mutInfo2['mutation'] in list(mutList.keys()):
                            mutList[mutInfo2['mutation']]['luad'] += 1
                        
                    genes[thisMutation.upper()]['luad'] += 1
                else:
                    genes['Other']['luad'] += 1
        else:
            if 'tcga' in sampleID.lower():
                if sampleID in tcgaIDs:
                    if thisMutation.upper() in geneKeys:
                        
                        mutInfo1, mutInfo2 = build_mutation(dataMutations[i,:], list(mutTypes.keys()), list(mutEff.keys()))
                    
                        tcgaDict[sampleID]['genes'][thisMutation.upper()].append(mutInfo1)
                        mutTypes[mutInfo1['type']]['tcga'] += 1
                        mutEff[mutInfo1['class']]['tcga'] += 1
                        if mutInfo1['mutation'] in list(mutList.keys()):
                            mutList[mutInfo1['mutation']]['tcga'] += 1
                        
                        if mutInfo2:
                            print(mutInfo2)
                            tcgaDict[sampleID]['genes'][thisMutation.upper()].append(mutInfo2)
                            mutTypes[mutInfo2['type']]['tcga'] += 1
                            mutEff[mutInfo2['class']]['tcga'] += 1
                            if mutInfo2['mutation'] in list(mutList.keys()):
                                mutList[mutInfo2['mutation']]['tcga'] += 1
                            
                        
                        genes[thisMutation.upper()]['tcga'] += 1
                    else:
                        genes['Other']['tcga'] += 1
    
    print( dataMutations[0,:] )
    
    ## Debug and Status output
    print('Number of LUAD cases: ' + str(numLuadCases))
    print('Number of Tcga cases: ' + str(numTcgaCases))
    print('Number of extra cases: ' + str(numExtraCases))
    print('-------------------------------------')
    
    print('\nGene Counts: ')
    print(genes)
    
    print('\nNumber of Genes: ')
    print(len(genes))
    
    print('-------------------------------------')
    
    print('\nCancer Type Counts: ')
    print(cancerClass)
    
    print('-------------------------------------')
    
    print('\nMutation Types: ')
    print(mutTypes)
    
    print('-------------------------------------')
    
    print('\nMutation Classes: ')
    print(mutEff)
    
    print('-------------------------------------')
    
    print('\nTotal Unique Mutations: ')
    print(len(mutList))
    
    print('-------------------------------------')
    
    print('\nMutation List: ')
    print(mutList)
    
    ## SAVE DATA
    pickle_1 = open(dataPath + "\genes.pickle", "wb")
    pickle.dump(genes, pickle_1)
    pickle_1.close()
    
    pickle_2 = open(dataPath + "\cancerClass.pickle", "wb")
    pickle.dump(cancerClass, pickle_2)
    pickle_2.close()
    
    pickle_3 = open(dataPath + "\mutationTypes.pickle", "wb")
    pickle.dump(mutTypes, pickle_3)
    pickle_3.close()
    
    pickle_4 = open(dataPath + "\mutationEffects.pickle", "wb")
    pickle.dump(mutEff, pickle_4)
    pickle_4.close()
    
    pickle_5 = open(dataPath + "\\tcgaData.pickle", "wb")
    pickle.dump(tcgaDict, pickle_5)
    pickle_5.close()
    
    pickle_6 = open(dataPath + "\luadData.pickle", "wb")
    pickle.dump(luadDict, pickle_6)
    pickle_6.close()
    
    pickle_7 = open(dataPath + "\mutationList.pickle", "wb")
    pickle.dump(mutList, pickle_7)
    pickle_7.close()
    return

if __name__ == '__main__':
    main()