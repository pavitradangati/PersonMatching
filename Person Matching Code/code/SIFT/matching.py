import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

probe = np.load('./probe_combined_sift.npy', allow_pickle = True)
gallery = np.load('./gallery_combined_sift.npy', allow_pickle = True)


probe_des = probe[:,0]
gallery_des = gallery[:,0]
print(probe_des.shape[0])

#probe_des[probe_des==None] = [-np.inf]
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
scores = np.zeros((probe_des.shape[0], gallery_des.shape[0]))

for i,des1 in tqdm(enumerate(probe_des)):
    for j, des2 in tqdm(enumerate(gallery_des)):
        if type(des2) == type(None) or type(des1) == type(None):
            scores[i][j]= -float("inf")
            continue
        #print(type(des1), type(des2))
        #print(des1.shape, des2.shape)
        matches = bf.knnMatch(des1, des2, k=2)
        #print(matches[0])
        good = []
        for m in matches:
            if len(m)<2:
                good.append(m[0])
                continue
            if m[0].distance < 0.7*m[1].distance:
                good.append(m[0])
        scores[i][j] = len(good)


np.save('./scores_sift.npy', scores)
scores_ = np.load('./scores_sift.npy', allow_pickle=True)
#Compute the rank of the scores using CMC
rankAcc = []
for rank in range(1,6):
    cmc = np.argsort(scores_, axis=1)[:,-rank:]
    accuracy = np.sum([1 for i in range(probe_des.shape[0]) if
                       probe[i,2] in gallery[cmc[i], 2]])/probe_des.shape[0]
    rankAcc.append(accuracy)

print(rankAcc)
#[print(probe[i,1], gallery[cmc[i], 1]) for i in range(10)]

#Plot the rank vs accuracy for SIFT features
plt.plot(np.arange(1,11), rankAcc, 'ro-', label = 'SIFT Accuracy')
plt.xlabel('Rank')
plt.ylabel('Accuracy')
plt.title('Cummalative Matching Characteristics')
plt.savefig('./AccuracySIFT.jpg')

