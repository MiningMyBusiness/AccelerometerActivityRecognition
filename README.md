# Accelerometer Activity Recognition
Classifying activities of daily living (ADL) from wrist-worn accelerometer readings.

## The data 
Many of us have Apple Watches or Fitbits with accelerometers built-in. A lot of that data is gathered by Apple, Fitbit, or other apps that pair with those devices (supposedly, anonymized). To see how much this data could reveal about a person's daily activities, I used this [dataset](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer) with 3-axis accelerometer readings from a wrist-worn accelerometer and labeled with the activity that the wearer was performing. 

These are the following activities found in the dataset (presented as labeled in the data, in alphabetical order):
1) Brush_teeth
2) Climb_stairs
3) Comb_hair
4) Descend_stairs
5) Drink_glass
6) Eat_meat
7) Eat_soup
8) Getup_bed
9) Liedown_bed
10) Pour_water
11) Sitdown_chair
12) Standup_chair
13) Use_telephone
14) Walk

Here are some plots of the data from the 3 axes of the accelerometer for each activity:

<img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Brush_teeth.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Climb_stairs.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Comb_hair.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Descend_stairs.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Drink_glass.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Eat_meat.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Eat_soup.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Getup_bed.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Liedown_bed.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Pour_water.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Sitdown_chair.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Standup_chair.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Use_telephone.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/Walk.png" width="400">

For more information about the data please visit the [repository](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer) for the data at the UCI Machine Learning database.

## The classifier
Using the raw data for each activity, I performed the following processing and machine learning tasks

1) split the data into 5 second time windows with 2 seconds of overlap - 160 time points by 3 axes for a total of 4954 time windows each with a label of the activity represented by that window.  

2) normalize the data so it ranges between 0-1. 

3) built a deep 1D-convolutional encoder-decoder neural net that compresses the 160 by 3 dimensional data into 128 latent dimensions
and then reconstructs the original signal of 160 by 3 dims. After 60 epochs of training (3 seconds each), the autoencoder could reconstruct each of the original signal axes with the following average R-squared values on a validation test set (0.95, 0.84, 0.90) - one R-squared value for each axis. So the autoencoder captures on average 90% of the information in the original 160 by 3 dimensional signal into a 128 dimensional vector.

4) built a classifier using the encoder from the encoder-decoder network with the addition of two densely connected layers for classification of the activity. The weights of the encoder were frozen and only the two densely connected layers were allowed to learn for 15 epochs.

5) the weights of the last two layers of the encoder were unfrozen and were allowed to be trained for 15 epochs with a slow learning rate.

The classifier has a final softmax layer which provides a probability that the specific time window belongs to one of the activities listed above. Refer to the [code](https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/blob/master/ClassifyActivity.py) for more details. 

## The results and conclusions
If we only pick the activity the classifier attributes the highest probability to, then the classifier has an overal accuracy of 70.1%. However, the actual activity is within the three activities predicted with the highest probability by the classifier 92.1% of the time.

Below is the confusion matrix for the classifier when the activity with the highest probability is picked.

<img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/ConfusionMatrix.png" width="600">

The labels on the y-axis of the confusion matrix denotes the true activity the user was performing while the labels on the x-axis are the activities predicted by the classifier. The color denotes the proportion of the true activity (each row) that was classified to be in that category. For instance, when the user was actually climbing stairs (2nd row), the classifier predicted the user was climbing stairs about half of the time and predicted that the user was walking for the other half. 

Some activities are clearly much easier to detect than others. For instance, Brush_teeth (first) and Eat_meat (6th) have high accuracy, precision, and recall. However, Liedown_bed (9th) has poor accuracy, precision, and recall. Interestingly, lying down in bed, getting up from bed, sitting down in a chair, and standing up from a chair are often confused for each other by the classifier. These could be differentiated more accurately with more information, like the time of day (people tend to lie down in bed at night) or the room the person is in. 

Some activities had great recall but poorer precision like walking (last) and pouring water (5th from last). Most instances of when the user was walking were classified by the model as walking but the model sometimes classified climbing or descending stairs as walking too, which is understandable. Similarly, almost all instances of when the user was pouring water was classified by the model as pouring water but it also sometimes confused eating soup for pouring water, which may also be understandable. 

On the other hand, there were some activities that had great precision but poorer recall. For instance, when the model predicted that the user was using the telephone, the user was indeed using the telephone. But sometimes the model missclassified some instances of using the telephone for combing hair or drinking from a glass, which is also understandable. 

## The scary part
Since this classifier outputs a probability that the user is doing some activity, the prediction of this classifier can be used in conjuction with a state transition model (Markov transition matrix), that describes the statistics of changing from one activity to another, to predict the most likely sequence of activities in real-time. Moreover, a more accurate inference can be drawn about the most likely sequence of activities by using a [forward-backward algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm) from a longer train of accelerometer data.

When this information is combined with the time-of-day and the GPS location (easily collected by cellphone) it becomes even more powerful. But how feasible is it to collect this data over the entire day without the user noticing a decrement in performance of their device? Let's see. 

There are 3 axes in the accelerometer with 6 bit resolution so 6 X 3 = 18 bits per sample. The data here is sampled at 32Hz so 32 X 18 = 576 bits/sec. That's 576 / 8 = 72 bytes/sec. This amount of info can easily streamed live in the background from an Apple Watch to the iPhone over bluetooth without the user even noticing a decrement in performance (Bluetooth 4.0 can go up to 250 Mbps). 

Accounting for the number of seconds in a day, the user will generate 72 X 60 X 60 X 24 = 6,220,800 bytes/day or 6.22 Mb over the course of an entire day. This can easily be saved on an iPhone without any appreciable loss of phone memory and sent over LTE or Wifi. The data can also be saved on the phone and uploaded to server in pieces over the course of the day.

Moreover, Apple and Fitbit have the resources to collect much larger datasets with their devices with more activities than the one used for this study (potentially far more embarassing and private activities). They probably know with a fair degree of accuracy what you're doing every minute of ever day. You should probably take it off when you don't need it. 

## Executing the scary part
We can expand this dataset in three ways to build a system that might resemble what Apple and Fitbit may need to actually keep track of your daily activites: 

1) We can stitch the time-series data for some of these activities in a sequence and see how well a activity classifier combined with a recursive Bayesian estimator can figure out the sequence in real-time as the sequence in presented and also after an entire sequence is provided. This can be used to compare accuracies of online activity analyses with offline analyses. 

2) We can superimpose the time-series of some of these activites to see how well the activity classifier deals with instances where someone was doing two activities at once, like brushing their teeth and walking at the same time.

3) We can add noise to the time-series data to see how accuracy is influenced by addition of noise. 

### Dealing with noise 
Let's start by doing the easiest of these and seeing how the classifier deals with noise. Since the classification is really being done with the encoded 128 dimensional data. Let's see how well the encoder-decoder network reconstructs a signal fed into it with added noise. 

<img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/RealSignal.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/NoiseSignal.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/RealSignal_recon.png" width="400"><img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/NoiseSignal_recon.png" width="400">

Above we can see a real signal and a noisy signal with added Gaussian noise with mean of 0 and standard deviation of 0.05. The reconstructions of the real and noisy signal from the encoder-decoder network look pretty similar. In fact, the reconstructions of the noisy signals from the encoder-decoder network have R-squared values of (0.93, 0.82, 0.89) for each axis across the validation set (variance weighted averaging) when compared to the real signals. These R-squared values are only slightly lower than those for the reconstructions of the real signals by the encoder-decoder network.

In fact, even if we increase the standard deviation of the added noise by 2X to 0.1 (an amount of noise a cellphone accelerometer would never have), the R-squared values of reconstructions from noisy signals are still (0.91, 0.78, 0.87). This is noticably lower but still agrees quite well with the R-squared values for the reconstructions of the real signals. Since R-squared values are a measure of the proportion of shared information between two signals, these results suggest that the encoder retains much of the original information in the real signal even in the face of very high noise. Therefore, the fact that the encoder network makes up the bottom of the classifier makes it more resilient to noisy inputs. 

### Dealing with superimposed activities 
Working with superimposed activities is tricky because we trained a classifier where the final layer was a softmax layer. So the layer's output sums to 1 where the neuron that is most likely to be the class of the input has the highest output value. This kind of network can solve a multi-class problem where we have more than 2 classes but not a multilabel problem where each input can belong to more than one class. 

Having superimposed activities makes this a multi-label problem. We can deal with superimposed activities by training a different network with a final sigmoid activation classification layer. And we may be able to generate the dataset to train the network by combining two or more activities that make sense. For instance, walking and using the telephone or descending stairs and combing hair. Note that some activities don't make any sense together, like sitting down in a chair and getting up from chair are mutually exclusive in time. 

However, simply adding the accelerometer signals from two different activities together will not work because we will produce unrealistic data that may sum to far greater than 6 bits. We can see that combing hair and brushing teeth are essentially maxed out on the range of the accelerometer. Maybe we can center our data and then superimpose signals and then limit range from 0 to 1. However, all of these changes will require us to retrain the encoder-decoder network and the classifier. Also, it's not clear how closely this generated dataset will resemble real data of superimposed activities. It's a safe bet to say that the accuracy of the multi-label classifier for each class will probably not be better than the accuracy of the multi-class classifier we already built. It may be better to build a superimposed activity classifier once we actually have that dataset...

### Predicting a sequence of actions
Before we can predict a sequence of actions with the multi-class classifier we've already built, we need to have data for a sequence of actions. Moreover, if we want to use the forward-backward algorithm or any kind of filtering technique for real-time sequence estimation, we need a state transition model for the system. The state transition model basically gives the probability of the user going from one activity to the next one. For instance, there is some probability that after walking, the user will start climbing stairs and maybe a similar probability that the user will descend stairs. However, it is very unlikely that the user will get-up from a chair after walking since they would already need to be seated in a chair which makes walking impossible. 

Inherent in this state transition model is the Markov assumption that the next activity of the user only depends upon the previous one. While this assumption may be an over-simplification since the last two or three activities may be even better able to predict the next one, in practice, the Markov assumption still works quite well when combined with a classifier and is much less complicated. Usually a state transition matrix is learned from data but we don't have data for a sequence of activities, so I'll make a state transition matrix that seems reasonable. 
