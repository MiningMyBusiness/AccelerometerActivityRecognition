## Read in text files for each activity and save the
##      data in the text files as a list of arrays
##
## Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

# grab all files in the directory
allFiles = glob.glob('*')

# get only the files that have data
dataFolders = []
for file in allFiles:
    if '.m' not in file and '.txt' not in file and '.py' not in file:
        dataFolders.append(file)

# read in data
for folder in dataFolders:
    dataFiles = glob.glob(folder + '/*')
    dataList = []
    for file in dataFiles:
        data = open(file, 'rb').read()
        data = str(data)
        data = data.split('\\r\\n')
        data = data[1:-1] # ignore first entry
        dataMat = np.zeros((len(data), 3))
        for i,entry in enumerate(data):
            entryList = entry.split(' ')
            for j,elem in enumerate(entryList):
                dataMat[i,j] = int(elem)
        dataList.append(dataMat)
    folderSave = folder + '/fullData.pkl'
    pickle.dump(dataList, open(folderSave, "wb"))
