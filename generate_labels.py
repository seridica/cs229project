# -*- coding: utf-8 -*-
"""
Count features
November 18 2017
CS229 Project

This function produces the labels

1) Binary labels - Ia,Ib vs. others
2) Multiclass labels - I, II, III, IV
3) Multiclass labels - Ia, Ib, IIa, IIb, IIIa, IIIb, IV

@author: Calvin
"""
import numpy as np
import pickle

def label_helper(dataIn, labelMap):
    
    # Generate Label Vector
    nDat = len( dataIn )
    labelVector = np.zeros((nDat, 1))
    dictKeys = list(dataIn.keys())
    for i in range(nDat):
        thisLabel = dataIn[dictKeys[i]]['grade']
        if (thisLabel in list( labelMap.keys() ) ):
            labelVector[i] = labelMap[thisLabel]
        else:
            labelVector[i] = np.inf
            
    return labelVector

def binary_labels(dataIn, zeroLabel=-1):
    
    # Labels
    labelMap = {'IA': zeroLabel,
                'IB': zeroLabel,
                'IIA': 1,
                'IIB': 1,
                'IIIA': 1,
                'IIIB': 1,
                'IV': 1
                }
    return label_helper( dataIn, labelMap )

def class_labels(dataIn):
    
    # Labels
    labelMap = {'IA': 0,
                 'IB': 0,
                 'IIA': 1,
                 'IIB': 1,
                 'IIIA': 2,
                 'IIIB': 2,
                 'IV': 3
                 }
    
    return label_helper( dataIn, labelMap )