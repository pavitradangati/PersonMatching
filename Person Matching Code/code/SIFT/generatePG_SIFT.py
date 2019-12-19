import numpy as np 
import pandas as pd 
feature_array = np.load('./faceFeaturesGT.npy', allow_pickle=True).reshape(-1,1)
indices_array = (pd.read_csv('./imageListGT.csv')).values
print(feature_array.shape, indices_array.shape)

print(indices_array[:10, :])
final_array = np.concatenate((feature_array, indices_array), axis = 1)
print(final_array.shape)

print(final_array[:10, 2])

people_ids = np.unique(final_array[:,2])
gallery = None
probe = None
gallery_features = None
gallery_indices = None
probe_features = None
probe_indices = None
for i in people_ids:
    count =np.where(final_array[:,2]==i)[0]
    lower = int(0.8*count.shape[0]) 
    gallery_idx = count[:lower]
    probe_idx = count[lower:]
    if gallery is None:
        gallery = final_array[gallery_idx, :]
        gallery_indices = final_array[gallery_idx, -2:]
    else:
        gallery = np.concatenate((gallery, final_array[gallery_idx, :]), axis=0)
        gallery_indices = np.concatenate(( gallery_indices, final_array[gallery_idx, -2:]), axis=0)
    if probe is None:
        probe = final_array[probe_idx, :]
        probe_indices = final_array[probe_idx, -2:]
    else:
        probe = np.concatenate((probe, final_array[probe_idx, :]), axis=0)
        probe_indices = np.concatenate(( probe_indices, final_array[probe_idx, -2:]), axis=0)

np.save('./gallery_combined_sift.npy', gallery)
np.save('./probe_combined_sift.npy', probe)
np.save('./gallery_list_sift.npy', gallery_indices)
np.save('./probe_list_sift.npy', probe_indices)
print(gallery.shape, probe.shape)
print(gallery_indices.shape, probe_indices.shape)
print('Done')
