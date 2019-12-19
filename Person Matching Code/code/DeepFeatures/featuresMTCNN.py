import numpy as np 
import pandas as pd 
feature_array = np.load('./faceFeatures.npy')
indices_array = (pd.read_csv('./imageList.csv')).values
final_array = np.concatenate((feature_array, indices_array), axis = 1)
people_ids = np.unique(final_array[:,513])
gallery = None
probe = None
for i in people_ids:
    count =np.where(final_array[:,513]==i)[0]
    newerIdx = map(lambda x: x if final_array[x,:512].all() not in [-np.Inf,
                                                                    np.Inf] else None, count)
    newerIdx = [x for x in newerIdx if x!=None]
    lower = int(0.8*len(newerIdx)) 
    gallery_idx = newerIdx[:lower]
    probe_idx = newerIdx[lower:]
    if gallery is None:
        gallery = final_array[gallery_idx, :]
    else:
        gallery = np.concatenate((gallery, final_array[gallery_idx, :]), axis=0)
    if probe is None:
        probe = final_array[probe_idx, :]
    else:
        probe = np.concatenate((probe, final_array[probe_idx, :]), axis=0)
np.save('./gallery_combined_cnn.npy', gallery)
np.save('./probe_combined_cnn.npy', probe)
print(gallery.shape, probe.shape)
print(np.max(gallery[:,:512]), np.min(gallery[:,:512]))
print(np.max(probe[:,:512]), np.min(probe[:,:512]))
print('Done')
