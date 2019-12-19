from PIL import Image
import os
import glob
from tqdm import tqdm
import pdb
import numpy as np
import sys
sys.path.append("../")
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import re
from facenet_pytorch.models.mtcnn import prewhiten
from torchvision.transforms import functional as F

runQuery = sys.argv[1]
# Get data path
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/')

# Check if datapath exists
if not os.path.isdir(data_path):
    raise Exception("X0X: Path Not Found")

# Get test and val data paths
# Currrently only test and val images are being used
celeb_path = os.path.join(data_path, 'img_align_celeba')
# Get all image filenames from test and val folders
celeb_img_filenames = os.listdir(celeb_path)

celeb_img_filenames = [os.path.join(celeb_path, l) for l in celeb_img_filenames if not l.startswith("._")]
query_path = os.path.join(data_path, 'celebTest')
# Get all image filenames from test and val folders
query_img_filenames = os.listdir(query_path)

query_img_filenames = [os.path.join(query_path, l) for l in query_img_filenames if not l.startswith("._")]
print("Num of images: ", len(celeb_img_filenames))
print("Num of images: ", len(query_img_filenames))


resnet = InceptionResnetV1(pretrained='vggface2').eval()
image_size = 160
id_ = 0
features = []
if runQuery  == 'False':
    # Get the ground truth bounding boxes
    with open('imageListCeleb.txt','w') as myfile:
        for im_path in tqdm(celeb_img_filenames):
            img = Image.open(im_path)
            img_name = re.findall(r"[\w']+", os.path.basename(im_path))[0]
            img_resized = img.resize((image_size, image_size), 2)

            img_resized = F.to_tensor(np.float32(img_resized))
            img_proc = prewhiten(img_resized)
            img_embedding = resnet(img_proc.unsqueeze(0)).cpu().data.numpy()
            features.append(img_embedding)
            myfile.writelines(im_path+'\n')
            id_ += 1
            if id_ > 60000:
                break
    np.save("faceFeaturesCeleb.npy", np.concatenate(features, axis=0))
else:
    # Get the ground truth bounding boxes
    with open('imageListCelebProbe.txt','w') as myfile:
        for im_path in tqdm(query_img_filenames):
            img = Image.open(im_path)
            img_name = re.findall(r"[\w']+", os.path.basename(im_path))[0]
            img_resized = img.resize((image_size, image_size), 2)

            img_resized = F.to_tensor(np.float32(img_resized))
            img_proc = prewhiten(img_resized)
            img_embedding = resnet(img_proc.unsqueeze(0)).cpu().data.numpy()
            features.append(img_embedding)
            myfile.writelines(im_path+'\n')
            id_ += 1
            if id_ > 60000:
                break
    np.save("faceFeaturesCelebProbe.npy", np.concatenate(features, axis=0))
