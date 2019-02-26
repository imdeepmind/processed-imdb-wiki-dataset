import numpy as np
import pandas as pd
import cv2

wiki = pd.read_csv('processedData/mat/wiki/wiki_meta.csv')

n = len(wiki)

paths = wiki['path'].values.reshape(1,n)[0]
details = wiki.drop(['path'], axis=1).values

BATCH_SIZE = 50000
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

        img = cv2.imread(path, 1)
        
        img = cv2.resize(img, (150,150))
        
        img = img.flatten()
        
        row = np.hstack((img, details[i]))
        
        data.append(row)
    
    columns = []
    for i in range(67500):
        columns.append('pixel' + str(i+1))
    
    columns = columns + ['age', 'gender']
        
    dataFrame = pd.DataFrame(data)

    dataFrame.columns = columns
    
    dataFrame.to_csv('processedData/images/wiki/wiki_'+str(batch)+'.csv')
    
    print('-- (BATCH-'+ str(batch+1) +') data save in csv file --')
    
    del start, end, data, path, img, row, columns, dataFrame
