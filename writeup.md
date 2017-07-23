# **Traffic Sign Recognition**

In this project, I've got to learn to use deep neural networks and convolutional neural networks to classify traffic signs.
Specifically, I've trained a model to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

With some preprocessing and data augmentation techniques, and some modification on LeNet architectrue, my final model get an accuracy of 98.8% on validation data
and an accuracy of 96.2% on test data.

In the new images prediction part, I collected 10 traffic signs from both google street and google picture, to test the performance of my model. Fortunate to see, it recognizes all the signs and get a 100% accuracy!

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/original.png "Original images"
[image3]: ./images/preprocessd.png "Preprocessed images"
[image4]: ./images/augmentation.png "Transform images"
[image5]: ./images/augmented_visualization.png " Augmented Visualization"
[image6]: ./test/image0.PNG "Traffic Sign 1"
[image7]: ./test/image1.png "Traffic Sign 2"
[image8]: ./test/image2.png "Traffic Sign 3"
[image9]: ./test/image3.png "Traffic Sign 4"
[image10]: ./test/image4.png "Traffic Sign 5"
[image11]: ./test/image5.png "Traffic Sign 6"
[image12]: ./test/image6.png "Traffic Sign 7"
[image13]: ./test/image7.png "Traffic Sign 8"
[image14]: ./test/image8.png "Traffic Sign 9"
[image15]: ./test/image9.PNG "Traffic Sign 10"


## Rubric Points

### Writeup / Submission

#### 1. Writeup

Here I will introduce the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

#### 2. Submission Files
You can find all project with submission files in this [Github Repository](https://github.com/demon221/CarND-Traffic-Sign-Classifier-P2.git).
You can also find the link of [project's code](https://github.com/demon221/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb) here.

-----
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy library to calculate summary statistics of the traffic.
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images for each class.
The number of images in each class is obviously asymmetrical.
In order to solve this, data augmentation shall be used to increase the mount of training images.

![alt text][image1]

----
### Design and Test a Model Architecture

#### 1. Preprocessed the Data Set
What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.
Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
(OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data.
Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


First let's see the visualization of one example for each individual class.
As we can see, the some of the original images have very different brightness values and possibly need to be normalized.

![alt text][image2]

Preprocess methods I have used:

1. Normalization to (-1, 1) - decrease the influence of the big value from images, make the back propagation and gradient decrease more efficient
2. Grayscale - reduce input data volume, helps to reduce training time, meanwhile the color makes almost no difference because of the similar colors and shapes between the traffic signs.
3. Equal histogram - in order to get a constant brightness level of all images

Here is the example of a traffic sign image after grayscaling.

![alt text][image3]

#### 2. Training Data Augmentation

The histogram of the number of images per classification category in the training data set is illustrating a huge variance.
To avoid our classifier biased towards the more frequent traffic signs, the training data shall be more equally distributed across the classes.
So I generate more fake data using image transform to let each class have the same image number.
This training data augmentation can also improve the accuracy to predict traffic sign in the real world. Because the real sign on the road is not so ideal.

Data augmentation techniques I have used:

1. random rotate -  -10 ~ +10 degree
2. random scale  -  0.8 ~ 1.2 scale
3. random translate - -2 ~ +2 pixel
4. random shear - -5 ~ +5 degree
5. random brightness - 0.5 ~ 2.0 factor

The different transform examples will be the augmentation data for training.
Here is an example of an original image and an augmented image.

![alt text][image4]

The bar chart shows the number of images for each class after data augmentation.
Every class now has 2500 training examples.
* The size of training set goes to 107500

![alt text][image5]

#### 3. Model Architecture
Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)
Consider including a diagram and/or table describing the final model.

My final model is modified based on the LeNet.
It is consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64    				|
| Flatten       		| flatten to 1600  								|
| Fully connected		| outputs 400   								|
| RELU					|												|
| Fully connected		| outputs 84       								|
| RELU					|												|
| Dropout				| Keep probability: 0.5		        			|
| Fully connected		| outputs 43       								|
| Softmax				|           									|


#### 4. Model Training.
Describe how you trained your model.
The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer and following hyper parameters:

* Batch size: 100
* Epochs：50
* Learning rate: 0.001
* Weight initial to truncated normal with mean = 0 and stddev = 0.1
* Dropout keep probability: 0.5

#### 5. Solution Approach
Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.
Include in the discussion the results on the training, validation and test sets and where in the code these were calculated.
Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps.
Perhaps your solution involved an already well known implementation or architecture.
In this case, discuss why you think the architecture is suitable for the current problem.

* I start the model with a standard LeNet architecture, and adjust it to RGB images and 43 classe logits.
I also add the dropout before the last output layer to improve the performance of over fitting.
Because the traffic sign recognition problem seems similar to the MNIST problem.

* Firstly the accuracy is only about 80% without any image preprocessing and augmentation. (1st)
The training data set is not so huge.
So I set epochs to 10 times and learning rate to 0.005 to make training faster, let the result feedback the accuracy result more quickly.

* Then I increase the accuracy to 92.7% with data preprocessing. (2nd, 3rd)

* I change the epochs to 50 times and decrease learning rate to 0.001.
Wish to train the model more times to evaluate its final performance. (4th)
But the accuracy seems not to become precise.

* Because that the training data has a huge asymmetrical in each class, I augment the data of each class to the same volume. (5th)
With this random generated training data and more equalized volume for each class, the accuracy increases to 94.2%.

* Finding the validation accuracy keeps mostly the same during last 10 - 20 epochs, I think to modify the LeNet architecture.
I try to increase the convolution layer sizes to feed the augmented training data. (6th, 7th)
After two time of adjusting, my final model goes to structure like this (C16 - C64 - FC400 - FC84 - SM43).
The final parameters are: Batchsize = 100, Epochs = 50, LearningRate = 0.001

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.8%
* test set accuracy of 96.2%

The solution approach is tracked as following:

###### 1st:
    - Standard LeNet5 (C6 - C16 - FC120 - FC84 - SM43)
    - Preprocessing: no
    - Batchsize = 128, Epochs = 10, LearningRate = 0.005
    - Validation Accuracy = 0.819
    - Test Accuracy = 0.799
###### 2nd:
    - Standard LeNet5 (C6 - C16 - FC120 - FC84 - SM43)
    - Preprocessing: normalization
    - Batchsize = 128, Epochs = 10, LearningRate = 0.005
    - Validation Accuracy = 0.922
    - Test Accuracy = 0.908
###### 3rd:
    - Standard LeNet5 (C6 - C16 - FC120 - FC84 - SM43)
    - Preprocessing: normalization, greyscale, equalhist
    - Batchsize = 128, Epochs = 10, LearningRate = 0.005
    - Validation Accuracy = 0.941
    - Test Accuracy = 0.927
###### 4th
    - Standard LeNet5 (C6 - C16 - FC120 - FC84 - SM43)
    - Preprocessing: normalization, greyscale, equalhist
    - Batchsize = 128, Epochs = 50, LearningRate = 0.001
    - Validation Accuracy = 0.952
    - Test Accuracy = 0.927
###### 5th
    - Standard LeNet5 (C6 - C16 - FC120 - FC84 - SM43)
    - Preprocessing: normalization, greyscale, equalhist
    - Data Augmentation: random transform to the same number for every class
    - Batchsize = 128, Epochs = 50, LearningRate = 0.001
    - Validation Accuracy = 0.970
    - Test Accuracy = 0.942
###### 6th
    - Standard LeNet5 (C32 - C64 - FC400 - FC84 - SM43)
    - Preprocessing: normalization, greyscale, equalhist
    - Data Augmentation: random transform to the same number for every class
    - Batchsize = 128, Epochs = 50, LearningRate = 0.001
    - Validation Accuracy = 0.977
    - Test Accuracy = 0.951
###### 7th
    - Standard LeNet5 (C16 - C64 - FC400 - FC84 - SM43)
    - Preprocessing: normalization, greyscale, equalhist
    - Data Augmentation: random transform to the same number for every class
    - Batchsize = 100, Epochs = 50, LearningRate = 0.001
    - Validation Accuracy = 0.988
    - Test Accuracy = 0.962


### Test a Model on New Images

#### 1. Acquiring New Images
Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web.
Some are from Google Street View and some are from google picture.

![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]  ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14]
![alt text][image15]

Ten traffic signs and their characters are explained here.
1. Priority road: the shape of the sign changes a little
2. Pedestrians: clear but other sign background
3. Stop: there is some noises and color faded on this sign
4. Roundabout mandatory: color faded on this sign
5. Keep right: the watch view has inclined which make recognize harder
6. Speed limit 50km/h: other characters on this sign
7. No passing for vehicles over 3.5 metric tons: tiny scale maybe hard to recognize
8. Speed limit 30km/h: this sign inclined to some degree
9. Right-of-way at the next intersection: sunlight makes this sign not equal light intensity
10. No entry: in the dawn time and with some slur maybe hard to recognize

#### 2. New Images Prediction
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction (Percentage)	        			| Result |
|:---------------------:|:--------------------------------------:|:-----------------:|
| Priority road      		| Priority road (100.00%)   	| Right
| Pedestrians  			    | Traffic signals (78.37%)			| Wrong
| Stop				        | Stop (100.00%)				|Right
| Roundabout mandatory	    | Roundabout mandatory (92.14%)	|Right
| Keep right		        | Keep right (100.00%)     		|Right
| Speed limit 50km/h		| Speed limit (50km/h) (100.00%) |Right
| No passing for vehicles over 3.5 metric tons		| No passing for vehicles over 3.5 metric tons (100.00%) |Right
| Speed limit 30km/h		| Speed limit (30km/h) (100.00%)  |Right
| Right-of-way at the next intersectio		| Right-of-way at the next intersection (100.00%)   |Right
| No entry		| No entry (100.00%)   						|Right

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%.

Only image 2 is predicted not correct. It may be because the sign of Pedestrians also has similar shape of triangle and similar backbround of tree. For these kind of similar signs, more training data with different samples shall be used.

It appears to have predicted the new signs almost the same to the  test accuracy. This is a good sign that the model performs well on real-world data as same as the test dataset. So the test data augmentation seems to have efficient influence. And while it's reasonable to assume that the accuracy would change when given more data points, judging by the low fidelity of a number of images in the training dataset. 

It's possible to choose some more critical signs to let the model predict. The correct rate will decrease for our training data still can not includes every worst cases in the real world.

#### 3. Top 5 Softmax Propabilities
Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top five soft max probabilities are shown in the following tables.

##### Image 1

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Priority road        |(100.00%)
|Speed limit (20km/h) |(0.00%)
|Speed limit (30km/h) |(0.00%)
|Speed limit (50km/h) |(0.00%)
|Speed limit (60km/h) |(0.00%)

##### Image 2

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Traffic signals |(78.37%)
|Road narrows on the right |(20.07%)
|Dangerous curve to the left |(1.43%)
|Pedestrians |(0.12%)
|Speed limit (120km/h) |(0.01%)

##### Image 3
| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Stop |(100.00%)
|Speed limit (20km/h) |(0.00%)
|Speed limit (30km/h) |(0.00%)
|Speed limit (50km/h) |(0.00%)
|Speed limit (60km/h) |(0.00%)
##### Image 4

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Roundabout mandatory |(83.71%)
|Speed limit (120km/h) |(12.04%)
|Speed limit (100km/h) |(3.30%)
|Children crossing |(0.70%)
|Priority road |(0.10%)

##### Image 5

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Keep right |(100.00%)
|Speed limit (20km/h) |(0.00%)
|Speed limit (30km/h) |(0.00%)
|Speed limit (50km/h) |(0.00%)
|Speed limit (60km/h) |(0.00%)

Image 6

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Speed limit (50km/h) |(97.34%)
|Speed limit (70km/h) |(2.66%)
|Speed limit (30km/h) |(0.00%)
|Keep right |(0.00%)
|Speed limit (120km/h) |(0.00%)

Image 7

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|No passing for vehicles over 3.5 metric tons |(100.00%)
|Speed limit (20km/h) |(0.00%)
|Speed limit (30km/h) |(0.00%)
|Speed limit (50km/h) |(0.00%)
|Speed limit (60km/h) |(0.00%)

Image 8

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Speed limit (30km/h) |(99.39%)
|Speed limit (120km/h) |(0.37%)
|Speed limit (50km/h) |(0.21%)
|Speed limit (100km/h) |(0.02%)
|Speed limit (70km/h) |(0.01%)

Image 9

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|Right-of-way at the next intersection |(100.00%)
|Speed limit (20km/h) |(0.00%)
|Speed limit (30km/h) |(0.00%)
|Speed limit (50km/h) |(0.00%)
|Speed limit (60km/h) |(0.00%)

Image 10

| Prediction    |     Probability	  	|
|:-------------:|:---------------------:|
|No entry |(100.00%)
|Speed limit (20km/h) |(0.00%)
|Speed limit (30km/h) |(0.00%)
|Speed limit (50km/h) |(0.00%)
|Speed limit (60km/h) |(0.00%)


Most images are almost predicted definetly. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

This question is a bit hard to resolve for me. Because the model architecture is not designed to analysis the weights and layers. Could you please give me some hints for this question so I can update it later.


