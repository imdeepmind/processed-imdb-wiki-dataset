import numpy as np
from scipy.io import loadmat
import pandas as pd

cols = ['dob', 'photo_taken', 'full_path', 'gender', 'name', 'face_score1', 'face_score2', 'imdb_id']

mat_file = "unprocessedData/mat/imdb.mat"

data = loadmat(mat_file)
data = data['imdb']

photo_taken = data[0][0][1][0]
full_path = data[0][0][2][0]
gender = data[0][0][3][0]
name = data[0][0][4][0]
face_score1 = data[0][0][6][0]
face_score2 = data[0][0][7][0]
celeb_id = data[0][0][9][0]

path = []
for file in full_path:
    path.append(file[0])

dob = []
for file in path:
    dob.append(file.split('_')[2])

names = []
for n in name:
    if len(n) > 0:
        names.append(n[0])
    else:
        names.append(np.nan)

genders = []
for n in range(len(gender)):
    if gender[n] == 1:
        genders.append('male')
    else:
	    genders.append('female')

theData = np.vstack((dob,photo_taken,path,genders,names,face_score1,face_score2, celeb_id)).T

dataFrame = pd.DataFrame(theData)
dataFrame.columns = cols

dataFrame.to_csv('processedData/mat/imdb/imdb_meta.csv')




