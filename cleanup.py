# -*- coding: utf-8 -*-
"""
Count features
November 18 2017
CS229 Project

This function cleans up input feature matrix and label vector (invalid data).
Other functions should represent invalid data with inf.

@author: Calvin
"""
import numpy as np

def cleanup_inputs(features, labels):
    
    # Make sure same number of data in each
    nFeat = features.shape[0]
    nLabel = labels.shape[0]
    
    assert( nFeat == nLabel )
    
    featureMatrix = np.copy(features)
    labelVector = np.copy(labels)
    
    # Remove invalids
    for i in range(nFeat):
        ind = nFeat - i - 1
        thisFeat = features[ind,:]
        thisLabel = labels[ind,:]
        
        # Check for invalid and remove
        if (np.inf in thisFeat or np.inf in thisLabel):
            featureMatrix = np.delete(featureMatrix, ind, 0)
            labelVector = np.delete(labelVector, ind, 0)
    
    return featureMatrix, labelVector