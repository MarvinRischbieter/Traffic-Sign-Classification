#**Traffic Sign Recognition**

Download the models here and copy to main folder:
* augmented.p https://goo.gl/jEgsrN
* preprocessed.p https://goo.gl/GspE6C

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: initial_data.jpg "Initial Data"
[image2]: ./writeup-images/grayscale_image.jpg "Grayscale Image"
[image3]: ./writeup-images/norm.jpg "Normalized Image"
[image4]: ./writeup-images/1.jpg "Sign 1"
[image5]: ./writeup-images/2.jpg "Sign 2"
[image6]: ./writeup-images/3.jpg "Sign 3"
[image7]: ./writeup-images/4.jpg "Sign 4"
[image8]: ./writeup-images/5.jpg "Sign 5"
[image9]: ./writeup-images/6.jpg "Sign 6"
[image10]: ./writeup-images/predict_1.jpg "Image 1 predictions"
[image11]: ./writeup-images/predict_2.jpg "Image 2 predictions"
[image12]: ./writeup-images/predict_3.jpg "Image 3 predictions"
[image13]: ./writeup-images/predict_4.jpg "Image 4 predictions"
[image14]: ./writeup-images/predict_5.jpg "Image 5 predictions"
[image15]: ./writeup-images/predict_6.jpg "Image 6 predictions"
[image16]: ./writeup-images/map.jpg "Map"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Pandas is the library that I used to get summary statistics of traffic
signs data sets. I got the following results:

| Data        		|     Result	        					|
|:---------------------:|:---------------------------------------------:|
| Training set        		| 34799   							|
| Validation set     	| 4410 	|
| Test set					|		12630										|
| shape of a traffic sign image	      	| (32, 32, 1)				|
| unique classes or labels in the data set	    |43 classes     									|




####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. There are 16 images from different classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided to convert the images to grayscale.

The following pictures are before and after grayscaling.

![alt text][image2]

After that I normalized the image data because it is more familiar for the net. It removes unnecessary data and leaves only the important part.

Here are some examples of normalized images:

![alt text][image3]

The difference between the original data set and the augmented data set is the fact that the augmented data is easier to understand and to predict because it has only the important data about the image, without any unnecessary data.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I created the Model with the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, out 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  out 16x16x64 				|
| Convolution 3x3	    | 2x2 stride,  out 16x16x64     									|
| Fully connected		| 2x2 stride,  out 16x16x64        									|
| Softmax				| 2x2 stride,  out 16x16x64        									|




####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For model training I used 25 EPOCHS with a  BATCH SIZE of 129.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]

The second and last images might be difficult to predict it is taken from a lower point.
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| Right-of-way     			| Right-of-way 										|
| Slippery road					| Slippery road											|
| Priority road	      		| Priority road					 				|
| No entry			| No entry      							|
| Road Work	      		| Road Work


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. I think that the images where kind of easy to predict but on the other way, I took random images. I didn't select them by any criteria.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the predictions for every image:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

As seen from the images, the predictions where 100% right.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image16]
