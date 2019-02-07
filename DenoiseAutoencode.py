## Read in lists of arrays for each activity
## Break up the continous data into time windows
## Build an encoder-decoder network to compress the time-series
## Tests the reconstruction of noisy signals from the
##  encoder-decoder network.
##
## Kiran Bhattacharyya

import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, UpSampling1D, AveragePooling1D, Flatten, Reshape, Dropout
from keras.models import Model
from keras import backend as K
import keras

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## grab all files in the directory
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Getting file names...')

allFiles = glob.glob('*')

# get only the files that have data
dataFolders = []
for file in allFiles:
    if '.m' not in file and '.txt' not in file and '.py' not in file and '.png' not in file:
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

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## break the time series data up into windows with some overlap
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Breaking time series into windows...')

windowSize = 5 # window size in seconds
skipSize = 3 # skip size for next window in seconds
maxLen = windowSize*32 # window size in points for 32Hz
step = skipSize*32 # step size in time points
activityWindows = np.zeros((1,maxLen,bigArray.shape[1]))
for i,bigArray in enumerate(bigDataList):
    print('Processing dataset '+str(i+1)+' of '+str(len(bigDataList)))
    activity = activityList[i]
    for j in range(0, len(bigArray) - maxLen, step):
        if i == 0 and j == 0:
            activityWindows[0,:,:] = bigArray[j:j+maxLen,:]
            labels = activity
        else:
            littleArray = np.reshape(bigArray[j:j+maxLen,:],
                                     (1, maxLen, bigArray.shape[1]))
            activityWindows = np.append(activityWindows,
                                        littleArray,
                                        axis=0)
            labels = np.append(labels, activity)

# convert all values between 0 - 1
activityWindows = activityWindows/64. #6-bit representation

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Build deep autoencoder
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Building the autoencoder model...')

encoded_dim = 128

input_data = Input(shape=(maxLen, bigArray.shape[1]))

x = Conv1D(100, 3, activation='relu', padding='same')(input_data)
x = AveragePooling1D()(x)
x = Conv1D(128, 3, activation='relu', padding='same')(x)
x = AveragePooling1D()(x)
x = Conv1D(100, 3, activation='relu', padding='same')(x)
x = AveragePooling1D()(x)
x = Conv1D(50, 3, activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
encoded = Dense(encoded_dim, activation='relu')(x)

# create encoder network
encoder = Model(input_data, encoded)

y = Dense(256, activation='relu')(encoded)
y = Dense(1000, activation='relu')(y)
y = Reshape((20,50))(y)
y = Conv1D(50, 3, activation='relu', padding='same')(y)
y = Conv1D(100, 3, activation='relu', padding='same')(y)
y = UpSampling1D()(y)
y = Conv1D(128, 3, activation='relu', padding='same')(y)
y = UpSampling1D()(y)
y = Conv1D(100, 3, activation='relu', padding='same')(y)
y = UpSampling1D()(y)
decoded = Conv1D(3, 3, activation='sigmoid', padding='same')(y)


# create autoencoder network
autoencoder = Model(input_data, decoded)

# define hyperparameters of autoencoder network
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Train the autoencoder model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# shuffle array and labels
rng_state = np.random.get_state()
np.random.shuffle(activityWindows)
np.random.set_state(rng_state)
np.random.shuffle(labels)
labels_cat = keras.utils.to_categorical(labels, num_classes)
nSamples = activityWindows.shape[0]
propTest = 0.2
nTrain = int((1-propTest)*nSamples)

autoencoder.fit(activityWindows[0:nTrain,:,:], activityWindows[0:nTrain,:,:],
                epochs=60,
                batch_size=32,
                shuffle=True,
                validation_data=(activityWindows[nTrain:,:,:], activityWindows[nTrain:,:,:]))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Validate autoencoder model
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Validating autoencoder model...')

from sklearn.metrics import r2_score
y_pred = autoencoder.predict(activityWindows[nTrain:,:,:])
dim1score = r2_score(activityWindows[nTrain:,:,0], y_pred[:,:,0],
        multioutput='variance_weighted')
dim2score = r2_score(activityWindows[nTrain:,:,1], y_pred[:,:,1],
        multioutput='variance_weighted')
dim3score = r2_score(activityWindows[nTrain:,:,2], y_pred[:,:,2],
        multioutput='variance_weighted')
print('R-squared for reconstruction of each axis')
print([dim1score, dim2score, dim3score])

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Add noise and test reconstruction
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Validating denoising ability of autoencoder...')

y_test = activityWindows[nTrain:,:,:]
y_test_noise = y_test + np.random.normal(0, 0.05, y_test.shape)
y_pred = autoencoder.predict(y_test)
y_pred_noise = autoencoder.predict(y_test_noise)

dim1score_noise = r2_score(y_test[:,:,0], y_pred_noise[:,:,0],
        multioutput='variance_weighted')
dim2score_noise = r2_score(y_test[:,:,1], y_pred_noise[:,:,1],
        multioutput='variance_weighted')
dim3score_noise = r2_score(y_test[:,:,2], y_pred_noise[:,:,2],
        multioutput='variance_weighted')
print('R-squared for reconstruction of each axis from noisy data')
print([dim1score_noise, dim2score_noise, dim3score_noise])

# plot real data
plt.plot(y_test[0,:,:])
plt.ylim((0,1))
plt.xlabel('Time (at 32Hz)')
plt.ylabel('Normalized acceleration')
plt.title('Real signal')
plt.tight_layout()
plt.savefig('RealSignal.png')
plt.close()

plt.plot(y_test_noise[0,:,:])
plt.ylim((0,1))
plt.xlabel('Time (at 32Hz)')
plt.ylabel('Normalized acceleration')
plt.title('Signal with noise')
plt.tight_layout()
plt.savefig('NoiseSignal.png')
plt.close()

plt.plot(y_pred[0,:,:])
plt.ylim((0,1))
plt.xlabel('Time (at 32Hz)')
plt.ylabel('Normalized acceleration')
plt.title('Reconstruction of real signal')
plt.tight_layout()
plt.savefig('RealSignal_recon.png')
plt.close()

plt.plot(y_pred_noise[0,:,:])
plt.ylim((0,1))
plt.xlabel('Time (at 32Hz)')
plt.ylabel('Normalized acceleration')
plt.title('Reconstruction of noisy signal')
plt.tight_layout()
plt.savefig('NoiseSignal_recon.png')
plt.close()
