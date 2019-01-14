import numpy as np
from scipy.io import loadmat
import pandas as pd

cols = ['dob', 'photo_taken', 'full_path', 'gender', 'name', 'face_score1', 'face_score2']

mat_file = "unprocessedData/mat/wiki.mat"

data = loadmat(mat_file)
data = data['wiki']

photo_taken = data[0][0][1][0]
full_path = data[0][0][2][0]
gender = data[0][0][3][0]
name = data[0][0][4][0]
face_score1 = data[0][0][6][0]
face_score2 = data[0][0][7][0]

path = []
for file in full_path:
    path.append(file[0])

dob = []
for file in path:
    dob.append(file.split('_')[1])

names = []
for n in name:
    if len(n) > 0:
        names.append(n[0])
    else:
        names.append(np.nan)

theData = np.vstack((dob,photo_taken,path,gender,names,face_score1,face_score2)).T

dataFrame = pd.DataFrame(theData)
dataFrame.columns = cols

dataFrame.to_csv('processedData/mat/wiki/wiki_meta.csv')





