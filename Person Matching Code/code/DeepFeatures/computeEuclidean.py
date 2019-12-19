# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:16:05 2019

@author: vaish
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
probe_combined = np.load('./probe_combined.npy', allow_pickle = True)
gallery_combined = np.load('./gallery_combined.npy', allow_pickle = True)

gallery_features = gallery_combined[:,:-2]
probe_features = probe_combined[:,:-2]

# Compute Euclidean distance
scoreList = []
#scores_Euclidean = np.linalg.norm((probe_features[:,:,None]-gallery_features[:,:,None].T), axis=1)
for i in tqdm(range(probe_features.shape[0])):
    diff = np.sqrt(np.array(np.sum((probe_features[i,:]-gallery_features)**2,
                          axis=1),dtype=np.float64))
    scoreList.append(diff/np.max(diff))
scores_Euclidean = np.asarray(scoreList)
print(scores_Euclidean.shape)
np.save('./scoresEuclidean1.npy',scores_Euclidean)
