# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:16:05 2019

@author: vaish
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

probe_combined = np.load('./probe_combined.npy', allow_pickle = True)
gallery_combined = np.load('./gallery_combined.npy', allow_pickle = True)
scoresPC = np.load('./scoresPC.npy', allow_pickle = True)
scoresEuc = np.load('./scoresEuclidean.npy', allow_pickle=True)
scores = np.load('./scores.npy', allow_pickle = True)

# Compute CMC rank 1
cmc1 = np.argmax(scores,axis=1)
print(cmc1.shape)

accuracy_rank1 = np.sum(probe_combined[:,513] == gallery_combined[cmc1,
                                                                  513])/probe_combined.shape[0]
print(probe_combined[0,512], gallery_combined[cmc1[0],512])
print('rank1',accuracy_rank1)
#Rank5
#rankAcc = [accuracy_rank1]
rankAcc = []
for rank in range(1,6):
    cmc = (np.argsort(scores, axis=1))[:,-rank:]
    accuracy = np.sum([1 for i in range(probe_combined.shape[0]) if probe_combined[i,513] in
                         gallery_combined[cmc[i], 513]])/probe_combined.shape[0]
    rankAcc.append(accuracy)
print(rankAcc)
[print(probe_combined[i,512], gallery_combined[cmc[i],512]) for i in
         range(8)]
"""
rankAcc = [0.6206563706563707, 0.6664092664092665, 0.6911196911196911,
           0.7065637065637066, 0.717953667953668, 0.7264478764478765,
           0.733011583011583, 0.7407335907335907, 0.7474903474903475,
           0.7534749034749034]

rankAccCNN = [0.4157226227257123, 0.48094747682801237, 0.5166495022313766,
              0.539993134225884, 0.5564709921043598, 0.5698592516306213,
              0.5798146240988672, 0.588053553038105, 0.5973223480947477,
              0.6035015447991761]
"""
plt.plot(np.arange(1,6), rankAcc, 'ro-', label='Ground Truth')
#plt.plot(np.arange(1,11), rankAccCNN, 'bo-', label='MTCNN')
plt.xlabel('Rank')
plt.ylabel('Accuracy')
#plt.legend()
plt.title('Cumulative Matching Characteristics')
plt.savefig('./Accuracy_GT_Cosine.jpg')

