# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_30_48_287.jpg "Driving in center"
[image3]: ./examples/right_2017_07_23_23_37_29_356.jpg "Recovery Image"
[image4]: ./examples/center_2016_12_01_13_39_21_759.jpg "Flipped Image #1"
[image5]: ./examples/center_2016_12_01_13_40_07_838.jpg "Flipped Image #2"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually 
and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Behavioral_Clonining_Project.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_IG.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the
 track by executing. Please note that the "carnd-term1" Anaconde package needs to be activated previously.
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The Behavioral_Clonining_Project.ipynb file contains the code for training and saving the convolution 
neural network. The file shows the pipeline I used for training and validating the model, and it contains 
comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a 
Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on track.
 Additionally, it was noticed that there are a lot of images with very small steering angles. Therefore, 
 only about 30% of the steering angles that were smaller than a steering threshold of "0.1" were accepted. The other images and their according information was disregarded. This was the final step that allowed my model to successfully run on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane 
driving, recovering from the left and right sides of the road. Since the vehicle drove off track in areas 
of dirt and sharp right curves, I recorded additional data to train better in these particular areas.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to run a model that works on the first meters 
of the track. Therefore, I used the Nvidia-model as a starting point. I thought this model is appropriate 
because it was advised in the lecture and a lot of brain power was already introduced into the model. 
Afterwards I reduced the model significantly with the goal to reduce the computational load. It turned 
out that a small neural network was running fine and I fine tuned the parameters from there.

The final step was to run the simulator to see how well the car was driving around track one. There were
 a few spots where the vehicle fell off the track in particular in sharp curves, and where the road
 border was not provided by yellow lines or red and white curbs. To improve the driving behavior in 
 these cases, I created additional data by recording recovery data in the particular areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving 
the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and 
layer sizes:

____________________________________________________________________________________________________
|Layer (type)|                     Output Shape|          Param #|     Connected to|                     
|---|---|---|---|
|lambda_1 (Lambda)|                (None, 32, 64, 1)|     0|           lambda_input_1[0][0]|             
|convolution2d_1 (Convolution2D)|  (None, 27, 59, 3)|     111|         lambda_1[0][0]|                   
|maxpooling2d_1 (MaxPooling2D)|    (None, 9, 19, 3)|      0|          convolution2d_1[0][0]|            
|convolution2d_2 (Convolution2D)|  (None, 7, 17, 3)|      84|			maxpooling2d_1[0][0]|             
|maxpooling2d_2 (MaxPooling2D)|    (None, 2, 5, 3)|       0|           convolution2d_2[0][0]|            
|dropout_1 (Dropout)|              (None, 2, 5, 3)|       0|           maxpooling2d_2[0][0]|             
|flatten_1 (Flatten)|              (None, 30)|            0|           dropout_1[0][0]|                  
|dense_1 (Dense)|                  (None, 1)|             31|          flatten_1[0][0]|                  
____________________________________________________________________________________________________
Total params: 226
Trainable params: 226
Non-trainable params: 0
____________________________________________________________________________________________________


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used the provided data set. Here is an example image of center 
lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so 
that the vehicle would learn to steer with larger magnitude when it is to far away from the center of the 
track. These images show what a recovery looks like starting.

![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would enlarge my data set 
and also reduce overfitting. For example, here is an image that has then been flipped:

![alt text][image4]

Additionally, I recorded data where the vehicle drives the track in the opposite direction in order to 
improve the steering.

![alt text][image5]

After the collection process, I had around 55000 number of data points, depending on how many data points 
were neglected because of the steering threshold. The neglection is done randomly, so that I cannot tell the exact number of data points. I then preprocessed this data by changing the color 
space to HSV and choosing the S channel. It proved in previous lessons to work very well and yield to 
distinguished features in this project. Furthermore, I resized the image to 32x64 (height and width in pixels) in order to reduce the computational load on my machine which lead to significantly quicker calculations. 
Within the model, I used a lambda layer to normalize the S-channel value.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was 
over or under fitting. 
The ideal number of epochs was 5 as evidenced by numerous experiments, where the validation loss stopped 
decreasing significantly after 5 epochs. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.