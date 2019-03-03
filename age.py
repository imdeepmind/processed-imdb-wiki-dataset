import pandas as pd
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split

meta = pd.read_csv('meta.csv')

meta = meta[meta['age'] >= 0]
meta = meta[meta['age'] <= 101]

meta = meta.values

D_train, D_test = train_test_split(meta, test_size=0.2, random_state=42)

paths = {'imdb':'imdb_crop', 'wiki':'wiki_crop/'}

for i in range(102):
    output_dir_train_male = 'dataset/age/train/' + str(i)
    output_dir_train_female = 'dataset/age/train/' + str(i)

    if not os.path.exists(output_dir_train_male):
        os.makedirs(output_dir_train_male)

    if not os.path.exists(output_dir_train_female):
        os.makedirs(output_dir_train_female)

    output_dir_test_male = 'dataset/age/test/' + str(i)
    output_dir_test_female = 'dataset/age/test/' + str(i)

    if not os.path.exists(output_dir_test_male):
        os.makedirs(output_dir_test_male)

    if not os.path.exists(output_dir_test_female):
        os.makedirs(output_dir_test_female)

counter = 0

for image in D_train:
    img = cv2.imread(image[2], 1)
    img = cv2.resize(img, (128,128))
    cv2.imwrite('dataset/age/train/' + str(image[0]) + '/' + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1

counter = 0

for image in D_test:
    img = cv2.imread(image[2], 1)
    img = cv2.resize(img, (128,128))
    cv2.imwrite('dataset/age/test/' + str(image[0]) + str(counter) + '.jpg', img)
    print('--('+str(counter)+')Processing--')
    counter += 1


