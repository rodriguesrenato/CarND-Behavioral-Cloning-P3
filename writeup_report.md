# **Behavioral Cloning**

---

## Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

> 1 . Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py 
  * containing the script to create and train the model
* drive.py 
  * for driving the car in autonomous mode
* model.h5 
  * containing a trained convolution neural network 
* writeup_report.md 
  * summarizing the results

> 2 . Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

> 3 . Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for preprocessing, training and validating the model, and it contains comments to explain how the code works.

There is also a jupyter notebook that has the same code, which makes easier to iterate through results and make new tests. It has some graph visualization of the results.

---

## Model Architecture and Training Strategy

> 1 . An appropriate model architecture has been employed

The model architecture chosen is based on the model published by the [autonomous vehicle team at NVIDIA](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), as shown in class, which gave the best results. I have tried LeNet architecture variations before without any considerable success. 

> 2 . Attempts to reduce overfitting in the model

The model architecture without dropout layers achieved great results and small overfitting, the car drove autonomously very well.

In the other hand, adding a ReLu activation followed by dropout regularization layer right before the output layer, it decreases even more the mse loss and reduces overfitting. (model.py lines TODO). 

The model was trained and validated on different data sets by spliting the training dataset on 80% for training and 20% for validation, to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

> 3 . Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

> 4 . Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of one lap of center lane driving and one lap of sinusoidal driving from the center to left and right sides of the road, simulating a recovery maneuvers to get back to the center of the road.

In addition to build the final dataset, the Center, Left and Right camera images were used and their respective augmented version (image fliped horizontally and negativated steerging angle), which increases the training dataset **six** times. The final size of the training dataset is 17100 images/measurements.

For details about how I created the training data, see the next section. 

---

## Model Architecture and Training Documentation

This section is explained directly in the `README.md` file, on the `Model Architecture and Training` section.