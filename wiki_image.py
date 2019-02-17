import numpy as np
import pandas as pd
import cv2

wiki = pd.read_csv('processedData/mat/wiki/wiki_meta.csv')
wiki = wiki.drop(['Unnamed: 0'], axis=1)

n = len(wiki)

paths = wiki['full_path'].values.reshape(1,n)[0]
scores = wiki['face_score1'].values.reshape(1,n)[0]
details = wiki.drop(['full_path', 'face_score1', 'face_score2', 'face0', 'face1', 'face2', 'face3'], axis=1).values

BATCH_SIZE = 10000
BATCH_NUMBER = 1
NO_BATCHES = (n // BATCH_SIZE) + 1
ELEMENTS = []

for i in range(NO_BATCHES):
    if BATCH_SIZE > (n - sum(ELEMENTS)):
        ELEMENTS.append((n - sum(ELEMENTS)))
    else:
        ELEMENTS.append(BATCH_SIZE)
        
for batch in range(NO_BATCHES):
    if batch < (BATCH_NUMBER-1):
        break
    
    start = sum(ELEMENTS[0:batch])
    end = sum(ELEMENTS[0:batch+1])
    
    data = []
    
    for i in range(start,end):
        path = 'unprocessedData/images/wiki_crop/' + paths[i]
    
        print('-- (BATCH-'+ str(batch+1) +')(' + str(i+1) + ') Currently processing image with path ' + path + ' --')

        haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

        img = cv2.imread(path, 0)

        face = haar_cascade_face.detectMultiScale(img, 1.1, 3)

        if len(face) > 0:
            for x,y,w,h in face:
                roi = img[y:y+h,x:x+w]
                
                face = cv2.resize(roi, (64,64))
        
                face = face.reshape(1,4096)
        
                row = np.hstack((face, details[i].reshape(1,4)))

                data.append(row[0])

                break
    
    columns = []
    for i in range(4096):
        columns.append('pixel' + str(i+1))
    
    columns = columns + ['dob', 'photo_taken', 'gender', 'name']
        
    dataFrame = pd.DataFrame(data)

    dataFrame.columns = columns
    
    dataFrame.to_csv('processedData/images/wiki/wiki_'+str(batch)+'.csv')
    
    print('-- (BATCH-'+ str(batch+1) +') data save in csv file --')
    
    del start, end, data, path, img, row, columns, dataFrame
