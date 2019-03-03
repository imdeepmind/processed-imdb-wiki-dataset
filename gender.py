import pandas as pd
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split

meta = pd.read_csv('meta.csv')

meta = meta.values

D_train, D_test = train_test_split(meta, test_size=0.2, random_state=42)

paths = {'imdb':'imdb_crop', 'wiki':'wiki_crop/'}

output_dir_train_male = 'dataset/gender/train/male'
output_dir_train_female = 'dataset/gender/train/female'

if not os.path.exists(output_dir_train_male):
    os.makedirs(output_dir_train_male)

if not os.path.exists(output_dir_train_female):
    os.makedirs(output_dir_train_female)

output_dir_test_male = 'dataset/gender/test/male'
output_dir_test_female = 'dataset/gender/test/female'

if not os.path.exists(output_dir_test_male):
    os.makedirs(output_dir_test_male)

if not os.path.exists(output_dir_test_female):
    os.makedirs(output_dir_test_female)


counter = 0

for image in D_train:
    img = cv2.imread(image[2], 1)
    img = cv2.resize(img, (128,128))
    if image[1] == 'male':
        cv2.imwrite('dataset/gender/train/male/' + str(counter) + '.jpg', img)
    else:
        cv2.imwrite('dataset/gender/train/female/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1


counter = 0

for image in D_test:
    img = cv2.imread(paths[image[3]] + '/' + image[2], 1)
    img = cv2.resize(img, (128,128))
    if image[1] == 'male':
        cv2.imwrite('dataset/gender/test/male/' + str(counter) + '.jpg', img)
    else:
        cv2.imwrite('dataset/gender/test/female/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1
