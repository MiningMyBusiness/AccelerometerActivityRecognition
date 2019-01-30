# AccelerometerActivityRecognition
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

For more information about the data please visit the [repository](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer) of the data.

## The classifier
Using the raw data for each activity I 

1) split the data in 5 second time windows - 160 time points by 3 axes for a total of 4954 time windows each with a label
2) built a deep 1D-convolutional encoder-decoder neural net that compresses the 160 by 3 dimensional data into 128 latent dimensions
and then reconstructs the original signal to 160 by 3. The autoencoder can reconstruct each of the original signal axis with the following average r-squared values on a validation test set (0.91, 0.76, 0.86).
3) built a classifier using the encoder with the addition of two densely connected layers. 

The classifier has a final softmax layer which provides a probability that the specific time window belongs to one of the activities listed above. If we only pick the activity the classifier attributes the highest probability to, then the classifier has an overal accuracy of 70%. However, if we pick the three activities with the highest probabilities to see if the actual activity is within those three, then the classifier achieves 91.5% overall accuracy. 

Refer to the code for more details. 

## The results and conclusions
Below is the confusion matrix for the classifier when I pick the activity with the highest probability.

<img src="https://github.com/MiningMyBusiness/AccelerometerActivityRecognition/raw/master/Figures/ConfusionMatrix.png" width="600">

Some activities are clearly much easier to detect than others. For instance, Brush_teeth and Eat_meat have high accuracy, precision, and recall. However, Liedown_bed has poor accuracy, precision, and recall. 

Interestingly, lying down in bed, getting up from bed, sitting down in a chair
