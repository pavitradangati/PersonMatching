import numpy as np 
import pandas as pd 
feature_array = np.load('./faceFeaturesGT.npy')
indices_array = (pd.read_csv('./imageListGT.csv')).values
print(feature_array.shape, indices_array.shape)
print(indices_array[:10, :])
final_array = np.concatenate((feature_array, indices_array), axis = 1)
print(final_array.shape)
print(final_array[:10, 512])
people_ids = np.unique(final_array[:,513])
#print(people_ids.shape, people_ids)
gallery = None
probe = None
gallery_features = None
gallery_indices = None
probe_features = None
probe_indices = None
for i in people_ids:
    count =np.where(final_array[:,513]==i))[0]
    lower = int(0.8*count.shape[0]) 
    gallery_idx = count[:lower]
    if all(x==np.Inf for x in final_array[count, :512]):
        continue
    probe_idx = count[lower:]
    if gallery is None:
        gallery = final_array[gallery_idx, :]
        gallery_features = final_array[gallery_idx, :-2]
        gallery_indices = final_array[gallery_idx, -2:]
    else:
        gallery = np.concatenate((gallery, final_array[gallery_idx, :]), axis=0)
        gallery_features = np.concatenate((gallery_features , final_array[gallery_idx, :-2]), axis=0)
        gallery_indices = np.concatenate(( gallery_indices, final_array[gallery_idx, -2:]), axis=0)
    if probe is None:
        probe = final_array[probe_idx, :]
        probe_features = final_array[probe_idx, :-2]
        probe_indices = final_array[probe_idx, -2:]
    else:
        probe = np.concatenate((probe, final_array[probe_idx, :]), axis=0)
        probe_features = np.concatenate((probe_features , final_array[probe_idx, :-2]), axis=0)
        probe_indices = np.concatenate(( probe_indices, final_array[probe_idx, -2:]), axis=0)

np.save('./gallery_combined.npy', gallery)
np.save('./probe_combined.npy', probe)
np.save('./gallery_features.npy', gallery_features)
np.save('./gallery_list.npy', gallery_indices)
np.save('./probe_features.npy', probe_features)
np.save('./probe_list.npy', probe_indices)
print(gallery.shape, probe.shape)
print(gallery_features.shape, probe_features.shape)
print(gallery_indices.shape, probe_indices.shape)
print('Done')
