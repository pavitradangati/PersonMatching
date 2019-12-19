# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:16:05 2019

@author: vaish
"""

import numpy as np
import pandas as pd
probe_combined = np.load('./probe_combined_cnn.npy', allow_pickle = True)
gallery_combined = np.load('./gallery_combined_cnn.npy', allow_pickle = True)

gallery_features = gallery_combined[:,:-2]
gallery_norm = gallery_features/np.linalg.norm(gallery_features)
probe_features = probe_combined[:,:-2]
probe_norm = probe_features/np.linalg.norm(probe_features)

# Compute cosine similarity

scores = np.dot(probe_norm, gallery_norm.T)
np.save('./scores_cnn.npy',scores)
print(scores.shape)
