import numpy as np
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta

cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

mat_file = "unprocessedData/mat/wiki.mat"

data = loadmat(mat_file)
data = data['wiki']

photo_taken = data[0][0][1][0]
full_path = data[0][0][2][0]
gender = data[0][0][3][0]
face_score1 = data[0][0][6][0]
face_score2 = data[0][0][7][0]

del data

path = []
for file in full_path:
    path.append(file[0])

genders = []
for n in range(len(gender)):
    if gender[n] == 1:
        genders.append('male')
    else:
        genders.append('female')

dob = []
for file in path:
    dob.append(file.split('_')[1])

age = []
for i in range(len(dob)):
    try:
        d1 = date.datetime.strptime(dob[i][0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(photo_taken[i]), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except:
        diff = -1
    age.append(diff)
    
    
del dob, photo_taken, full_path, gender

theData = np.vstack((age, genders, path, face_score1, face_score2)).T

dataFrame = pd.DataFrame(theData)
dataFrame.columns = cols

dataFrame = dataFrame[dataFrame['face_score2'] == 'nan']
dataFrame = dataFrame[dataFrame['face_score1'] != '-inf']

dataFrame = dataFrame.drop(['face_score1', 'face_score2'], axis=1)

dataFrame.to_csv('processedData/mat/wiki/wiki_meta.csv', index=False)
