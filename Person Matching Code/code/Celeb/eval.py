# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:16:05 2019

@author: vaish
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

scores = np.load('./scores_celeb.npy', allow_pickle = True)

# Compute CMC rank 1
cmc1 = np.argmax(scores,axis=1)
print(cmc1.shape)

#rint(probe_combined[0,512], gallery_combined[cmc1[0],512])
#Compute CMC Rank5
cmc5 = (np.argsort(scores, axis=1))[:,-5:]
#   [print(probe_combined[i,512], gallery_combined[cmc[i],512]) for i in
#         range(5)]

# Get the indices for top 5 matches which is the image ID
print(cmc5)
