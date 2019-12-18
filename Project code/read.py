import numpy as np 
import pandas as pd 
gallery = np.load('./gallery_features.npy', allow_pickle=True)
probe = np.load('./probe_features.npy', allow_pickle=True)
print(gallery.shape, probe.shape)