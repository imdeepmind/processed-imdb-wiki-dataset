import numpy as np
from scipy.io import loadmat
import pandas as pd

cols = ['dob', 'photo_taken', 'full_path', 'gender', 'face_score']

path = "imdb.mat"

data = loadmat(path)
data = data['imdb']

dob = data[0][0][0][0]
photo_taken = data[0][0][1][0]
full_path = data[0][0][2][0]
gender = data[0][0][3][0]
face_score = data[0][0][6][0]

path = []
for p in full_path:
    path.append(p[0])

theData = np.vstack((dob,photo_taken,path,gender, face_score)).T

dataFrame = pd.DataFrame(theData)
dataFrame.columns = cols

dataFrame.to_csv('imdb_meta.csv')