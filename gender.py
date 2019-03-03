import pandas as pd
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split

imdb = pd.read_csv('processedData/mat/imdb/imdb_meta.csv')
imdb['media_type'] = 'imdb'

wiki = pd.read_csv('processedData/mat/wiki/wiki_meta.csv')
wiki['media_type'] = 'wiki'

data = pd.concat((imdb, wiki))
data = data.values

D_train, D_test = train_test_split(data, test_size=0.2, random_state=42)

paths = {'imdb':'unprocessedData/images/imdb_crop', 'wiki':'unprocessedData/images/wiki_crop/'}

output_dir_train_male = 'processedData/images/train/male'
output_dir_train_female = 'processedData/images/train/female'

if not os.path.exists(output_dir_train_male):
    os.makedirs(output_dir_train_male)

if not os.path.exists(output_dir_train_female):
    os.makedirs(output_dir_train_female)

output_dir_test_male = 'processedData/images/test/male'
output_dir_test_female = 'processedData/images/test/female'

if not os.path.exists(output_dir_test_male):
    os.makedirs(output_dir_test_male)

if not os.path.exists(output_dir_test_female):
    os.makedirs(output_dir_test_female)

counter = 0

for image in D_train:
    img = cv2.imread(paths[image[3]] + '/' + image[2], 1)
    img = cv2.resize(img, (128,128))
    if image[1] == 'male':
        cv2.imwrite('processedData/images/train/male/' + str(counter) + '.jpg', img)
    else:
        cv2.imwrite('processedData/images/train/female/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1

counter = 0

for image in D_test:
    img = cv2.imread(paths[image[3]] + '/' + image[2], 1)
    img = cv2.resize(img, (128,128))
    if image[1] == 'male':
        cv2.imwrite('processedData/images/test/male/' + str(counter) + '.jpg', img)
    else:
        cv2.imwrite('processedData/images/test/female/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Image Processing--')
    counter += 1
