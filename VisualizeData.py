## Read in lists of arrays for each activity
## Make representative plots of accelerometer readings for each activity
##
## Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## grab all files in the directory
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Getting file names...')

allFiles = glob.glob('*')

# get only the files that have data
dataFolders = []
for file in allFiles:
    if '.m' not in file and '.txt' not in file and '.py' not in file:
        dataFolders.append(file)

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## create activity dictionary with numerical labels
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Creating activity dictionary...')

activityDict = {} # dictionaries to store activity string and label
activityDict_rev = {}
activityIndx = 0
for folder in dataFolders:
    if '_MODEL' in folder:
        folder = folder[:-6]
    if folder not in activityDict:
        activityDict[folder] = activityIndx
        activityDict_rev[activityIndx] = folder
        activityIndx += 1

num_classes = len(activityDict) # number of activities

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## read in data
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Reading in all data...')

activityList = [] # list of activities corresponding to data
bigDataList = [] # list of arrays with time series of accelerometer
for folder in dataFolders:
    folderLoad = folder + '/fullData.pkl'
    dataList = pickle.load(open(folderLoad, "rb"))
    for i,array in enumerate(dataList):
        if i == 0:
            bigArray = array
        else:
            bigArray = np.append(bigArray, array, axis=0)
    bigDataList.append(bigArray)
    if '_MODEL' in folder:
        folder = folder[:-6]
    activityList.append(activityDict[folder])

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Plot raw data
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Plotting raw data...')

lastLabel = 53 # assign arbitrary last label
for i,label in enumerate(activityList):
    if label != lastLabel:
        plt.plot(bigDataList[i][0:1000,0])
        plt.plot(bigDataList[i][0:1000,1])
        plt.plot(bigDataList[i][0:1000,2])
        plt.ylim((0,64))
        plt.xlabel('Time (at 32Hz)')
        plt.ylabel('Accelerometer reading (6 bits)')
        plt.title(activityDict_rev[activityList[i]])
        plt.tight_layout()
        plt.savefig('Figures/'+activityDict_rev[activityList[i]]+'.png')
        plt.close()
    lastLabel = label
