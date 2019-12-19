from PIL import Image
import os
import glob
from tqdm import tqdm
import pdb
import numpy as np
import sys
sys.path.append('../../../../')
import pandas as pd
import re
#from facenet_pytorch import MTCNN, InceptionResnetV1
from PersonMatching import MTCNN, InceptionResnetV1
from PersonMatching.models.mtcnn import prewhiten
from torchvision.transforms import functional as F

useGT = sys.argv[1]
visualFolder = 'visuals/' if not useGT else 'visualsGT'
# Get data path
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/')
cropped_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            visualFolder)

if not os.path.isdir(cropped_path):
    os.makedirs(cropped_path)
# Check if datapath exists
if not os.path.isdir(data_path):
    raise Exception("X0X: Path Not Found")

# Get test and val data paths
# Currrently only test and val images are being used
test_path = os.path.join(data_path, 'test')
val_path = os.path.join(data_path, 'val')

# Get all image filenames from test and val folders
# test_img_filenames = glob.glob(os.path.join(test_path, '*.jpg'))
# val_img_filenames = glob.glob(os.path.join(val_path, '*.jpg'))
test_img_filenames = os.listdir(test_path)
val_img_filenames = os.listdir(val_path)

test_img_filenames = [os.path.join(test_path, l) for l in test_img_filenames if not l.startswith("._")]
val_img_filenames = [os.path.join(val_path, l) for l in val_img_filenames if not l.startswith("._")]

print("Num of test images: ", len(test_img_filenames))
print("Num of val images: ", len(val_img_filenames))


resnet = InceptionResnetV1(pretrained='vggface2').eval()
image_size = 160
features = []
df = pd.read_csv(os.path.join(data_path, 'annotations/index.txt'), sep = " ", header=None)
df.columns = ['photoset_id', 'photo_id', 'xmin', 'ymin', 'width', 'height', 'identity_id', 'subset_id']
if useGT == 'False':
    mtcnn = MTCNN(image_size=image_size, margin=0)
    print('Entered')
    with open('imageList.txt', 'w') as myfile:
        for im_path in tqdm(val_img_filenames+test_img_filenames):
            img = Image.open(im_path)
            img_name = re.findall(r"[\w']+", os.path.basename(im_path))[0]

            im_name1, im_name2 = img_name.split("_")

            id_details = df.loc[df['photoset_id'] ==
                                int(im_name1)].loc[df['photo_id'] ==
                                                   int(im_name2)]
            for id_, row in id_details.iterrows():
                out  = row[['xmin', 'ymin', 'width',
                               'height','identity_id']].values.tolist()
            # Get cropped and prewhitened image tensor
            img_cropped = mtcnn(img, save_path=os.path.join(cropped_path,
                                                            os.path.basename(im_path)
                                                           ))
            # Calculate embedding (unsqueeze to add batch dimension)
            if img_cropped is None:
                img_embedding = -np.inf*np.ones((1,512))
            else:
                img_embedding = resnet(img_cropped.unsqueeze(0)).cpu().data.numpy()
            features.append(img_embedding)
            myfile.writelines(im_path+','+str(out[4])+'\n')
    np.save("./faceFeatures.npy", np.concatenate(features, axis=0))
else:
    # Get the ground truth bounding boxes
    with open('imageListGT.txt','w') as myfile:
        for im_path in tqdm(val_img_filenames+test_img_filenames):
            img = Image.open(im_path)
            img_name = re.findall(r"[\w']+", os.path.basename(im_path))[0]

            im_name1, im_name2 = img_name.split("_")

            id_details = df.loc[df['photoset_id'] ==
                                int(im_name1)].loc[df['photo_id'] ==
                                                   int(im_name2)]
            for id_, row in id_details.iterrows():
                out  = row[['xmin', 'ymin', 'width',
                               'height','identity_id']].values.tolist()
            #print(id_details)

                img_cropped = img.crop((out[0],
                                    out[1],out[0]+out[2],out[1]+out[3]))
                #print(img_cropped.size)
                img_resized = img_cropped.resize((image_size, image_size), 2)

                img_resized = F.to_tensor(np.float32(img_resized))
                img_proc = prewhiten(img_resized)
                img_embedding = resnet(img_proc.unsqueeze(0)).cpu().data.numpy()
                features.append(img_embedding)
                myfile.writelines(im_path+','+str(out[4])+'\n')
    np.save("faceFeaturesGT.npy", np.concatenate(features, axis=0))
