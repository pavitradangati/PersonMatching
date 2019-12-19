import numpy as np
import pandas as pd

# To compute the Cosine similarity between query image
# and imgaes in celebrity dataset

probe_combined = np.load('./faceFeaturesCelebProbe.npy', allow_pickle = True)
gallery_combined = np.load('./faceFeaturesCeleb.npy', allow_pickle = True)

gallery_features = np.concatenate((probe_combined[-2:,:],gallery_combined[:,:]),
                                  axis=0)
gallery_norm = gallery_features/np.linalg.norm(gallery_features)
probe_features = probe_combined[:-2,:]
print(probe_features.shape)
print(gallery_features.shape)
probe_norm = probe_features/np.linalg.norm(probe_features)

# Compute cosine similarity

scores = np.dot(probe_norm, gallery_norm.T)
np.save('./scores_celeb.npy',scores)
print(scores.shape)
