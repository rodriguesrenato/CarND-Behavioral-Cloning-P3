# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![](images/simulator.png)

This repository contains my implementation for the Behavioral Cloning Project of Udacity Self-Driving Car NanoDegree. Project instructions can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3).

In this project, a deep neural network was trained to clone driving behaviors and drive autonomously a car in the Udacity Simulator. The model was trained, validated and tested using Keras. The model outputs a steering angle to an autonomous vehicle.

Udacity has provided a simulator where you can drive a car around a track for data collection. Image data and steering angles were used to train a neural network and then this model is used to drive the car autonomously around the track.

This is the result of the trained model driving autonomously the car in the simulation:

![](run17.gif)

This project contains the following files:
| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
|  `model.py`             | Script used to create and train the model        |
|  `training_model.ipynb` | Notebook version to create and train the model   |
|  `drive.py`             | Script to drive the car in the Udacity simulator |
|  `model.h5`             | The trained Keras model                          |
|  `writeup_report.md`    | A report writeup file                            |
|  `run17.mp4`            | The video recording the vehicle driving autonomously around the track for one full lap |

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior 
- Design, train and validate a model that predicts a steering angle from image data
- Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
- Summarize the results with a written report

# Dependencies 

This project was built and was run on Ubuntu 18.04.5 LTS. An Anaconda (Conda) environment was used to help match the required package versions to run the same code in the Udacity workspace and along with the Udacity Simulator. 

To help setting up the environment and packages, `conda_requirements.txt` and `pip_requirements.txt` were provided.

The base dependencies/packages are required before installing other dependencies through `*_requirements.txt` files:
- Anaconda 4.10.1
- [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) - Term 1, Version 2 Linux

# Installation

Clone this repository in the desired folder on your computer. Consider all commands shown below are run in this folder when not specified.

```
cd ~/your_desired_folder
git clone https://github.com/rodriguesrenato/CarND-Behavioral-Cloning-P3.git
```
Create a new conda environment from `conda_requirements.txt` file:

```
conda create --name car-nd-p3 --file conda_requirements.txt
```

Activate your conda environment:

```
conda activate car-nd-p3
```

Install all required packages through pip:

```
pip install -r pip_requirements.txt 
```

NOTE: If you got problems configuring and installing the environment/packages, follow the instructions provided by Udacity [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md).

To install Tensorflow with GPU, follow this article [here](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc). You can create a new environment with the previous steps or install tensorflow-gpu along with the current conda environment (This environment uses tensorflow 1.10.0, so the GPU version has to be the same version as CPU version)

```
conda install tensorflow-gpu==1.10.0
```

To install the Udacity Simulator, simply download the `Term 1 -> Version 2 -> Linux` [here](https://github.com/udacity/self-driving-car-sim), unzip the `beta_simulator_linux` in this project folder, make `beta_simulator.x86_64` executable (`chmod +x beta_simulator.x86_64`, if your system is 64 bits).

# Usage

## Simulator

To launch the Simulator:

```
conda activate car-nd-p3
./beta_simulator_linux/beta_simulator.x86_64
```

When you open for the first time, you will have to select Screen Resolution of 640 x 480 and Graphics Quality as Fastest. 

After the simulator fully initializes, you can choose between `training mode` and `autonomous mode`. In `training mode` you can drive the car on the selected track in the menu and record driving data. In `autonomous mode`, you have to have `drive.py` running to drive the car autonomously on the road.

## Autonomous driving with `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. It can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### Create a video of the saved autonomous driving with `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

To generate a GIF version of the video using ffmpeg:

```
ffmpeg -ss 70 -t 40 -i run17.mp4 -vf "fps=30,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 run17.gif
```

The `-ss` will skip 70 frames, `-t` will use the next 40 seconds after skipping, `fps` will set the new fps for the gif (in this case, `fps=30` was chosen to generate a gif 2 time faster than video)

# Model Architecture and Training

## Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model (only a 100 Dense layer) to do a full test of autonomous driving in the Simulator, and after that, start testing more complex models. 

Before starting designing models, I collected more samples to get a better dataset, as explained in the next sections. To expand the dataset, the 3 given images per sample and the augmented version (image/measurement flipped) of them were used. I built a summary of the dataset to check if the samples are biased and well distributed.

Then, I have implemented and tested a model similar to LeNet-5 architecture, with 3 convolutional layers and two dense layers. This model performed poorly on the simulator. I thought this model might be appropriate because 3 convolutional layers would be able to detect enough higher level features to continue through the dense layers.

After that, I decided to implement a new model based on the model published by the [autonomous vehicle team at NVIDIA](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). This is a more complex model, with 5 convolutional layers, which will be able to detect even higher level features, followed by 4 dense layers. This model worked pretty well and the car succeeded in driving autonomously.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation dataset with a proportion of 80% and 20% respectively. I found that the current model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I tried to apply regularization by adding Dropout layers between the dense layers. I tried multiple configurations and rates. The configuration that worked the best was only one dropout layer between the two last dense layers, with a lower rate of 0.2.
 
The final step was to run the simulator to see how well the car was driving around track one. With the first designed models, the vehicle fell off the track and wasn't able to recover when it got off the center of the lane. With the model based on LeNeT, the car couldn't recover from sharp turns. The final model, the one based on the NVIDIA model, worked well, the car could recover when it got off the center and drive through the whole lap without getting off the road.

To improve the driving behavior in these cases, I have recorded a new full lap driving sinusoidally on the center of the road, which improves the model in the following training sessions.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

## Final Model Architecture

The final model architecture (model.py lines 112-141) are described in the table below:

| Layer (type)              | Description | Output Shape          | Param #|
|---------------------------|-------------|-----------------------|--------|
| lambda_5 (Lambda)         | Image Normalization  | (None, 160, 320, 3)   | 0      |
| cropping2d_5 (Cropping2D) | Image Cropping on ROI | (None, 65, 320, 3)    | 0      |
| conv2d_21 (Conv2D)        | Convolutional 5x5 filter, 2x2 stride, ReLu activation and 24 output filters  | (None, 31, 158, 24)   | 1824   |
| conv2d_22 (Conv2D)        | Convolutional 5x5 filter, 2x2 stride, ReLu activation and 36 output filters | (None, 14, 77, 36)    | 21636  |
| conv2d_23 (Conv2D)        | Convolutional 5x5 filter, 2x2 stride, ReLu activation and 48 output filters | (None, 5, 37, 48)     | 43248  |
| conv2d_24 (Conv2D)        | Convolutional 3x3 filter, 1x1 stride, ReLu activation and 64 output filters | (None, 3, 35, 64)     | 27712  |
| conv2d_25 (Conv2D)        | Convolutional 3x3 filter, 1x1 stride, ReLu activation and 64 output filters | (None, 1, 33, 64)     | 36928  |
| flatten_5 (Flatten)       | Flatten last convolutional output | (None, 2112)          | 0      |
| dense_17 (Dense)          | Fully connected with size 100, ReLu activation | (None, 100)           | 211300 |
| dense_18 (Dense)          | Fully connected with size 50, ReLu activation | (None, 50)            | 5050   |
| dense_19 (Dense)          | Fully connected with size 10, ReLu activation | (None, 10)            | 510    |
| dropout_21 (Dropout)      | Regularization Dropout of 0.2 | (None, 10)            | 0    |
| dense_20 (Dense)          | Output layer - predicted steering angle | (None, 1)             | 11     |

Parameters statistics:
- Total params: 348,219
- Trainable params: 348,219
- Non-trainable params: 0

My model chosen consists of a sequence of 5 convolutional neural network layers, followed by 4 fully connected layers. The first 3 convolutional layers have 5x5 filter sizes, 2x2 strides sizes and depths of 24, 36 and 48. The next 2 convolutional layers have 3x3 filter sizes and depth of 64. The following 4 fully connected layers have sizes of 100, 50, 10 and 1, as the last layer is the output layer. A dropout layer was added between the last two dense layers. (model.py lines 140) 

Each layer in the model includes RELU activation to introduce nonlinearity, except in the output layer, and the data is normalized from -0.5 to 0.5 in the model using a Keras lambda layer (model.py line 125). 

Here is a visualization of the architecture generated by keras util `plot_model`:

![alt text](images/model_vis.png)

## Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one. On the first three laps, I drive the vehicle as close as possible to the center lane. Here is an example image of center lane driving:

![alt text](images/lane_center.jpg)

On the fourth lap, I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to get back to the center of theroad. This gif image shows what a recovery looks like starting from center and getting back to center:

![alt text](images/recover_lap2.gif)

The first 3 laps (centered driving) were recorded and save separated from the last lap (recovering). After the collection process, there are 5193 data points (samples). 

I decided to use the 3 images from the Center, Left and Right camera of each sample. I added a `steering_correction` value to the steering angle of each corresponding image (the Center image has a correction value of `0`). The side camera's images could be interpreted as the central image of a car that isn't in the center of the lane, so that is why I need to add a correction value to the recorded steering angle because it will stimulate the car to drive back to the center when it gets to that position. These are examples of the Left, Center and Right camera output for the same sample set.

| Left | Center | Right |
|------|--------|-------|
|![alt text](images/sample_left.jpg) | ![alt text](images/sample_center.jpg) | ![alt text](images/sample_right.jpg)|
| steering_correction = 0.2 | steering_correction = 0 | steering_correction = -0.2 |

To augment the dataset, I flipped the images and steering angles of each sample image (center, left and right). This would duplicate the dataset, and also add more data of right turns to help model don't bias to left and the result would be reasonable, the same as if we have recorded this driving on a horizontally flipped track. For example, here is an image and the corresponding augmented one:

| Normal | Augmented (Flipped) |
|------|--------|
|![alt text](images/lane_center_normal.jpg) | ![alt text](images/lane_center_aug.jpg) |

The dataset expansion process is done before training. When the code load the recorded data, it will add to the sample dataset the six possible data from that recorded line in the .csv file, with the following format: [`image_path`,`steer_angle_corrected`,`augmented_flag`]

- `image_path`: 3 possibilities -  Center[0], Left[1] and Right[2]
- `steer_angle_corrected`: the measured steering angle plus the respective steering_correction, accordingly to the current image (center, left or right)
- `augmented_flag`: A boolean flag to indicate when this image have to be flipped or not (this flag will be used in the generator)

This samples dataset is randomly shuffled and split in 80% for the `train_samples` set and 20% for the `validation_samples` set

At this point, each sample data will generate 6 images/measurements, resulting in a training dataset with size of `24926` and validation dataset with size of `6232`.

Here is the histogram of the steering angles in each dataset:

![](images/dataset_distribution.png)

The histogram helps to check if training and validation dataset have similar distributions.

The generator function, which is responsible to read the images and augment the image/steering_angle when `augmented_flag` is set, returning them in batches. Using a generator to load and process the images avoids the program having to use a high amount of memory at once.

The images are pre processed directly in the Keras sequential model. The images are normalized from -0.5 to 0.5 through a lambda function and then cropped from top and bottom through Keras Cropping2D function to crop image on the lane area.

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was `4` as evidenced by the following graph, showing that at this point both curves get closer and stable before overfitting again.

![](images/training_mse_10.png)

When training the model again (running the whole model.py script) with lower epochs, this is the result of the new trained model:

![](images/training_mse_4.png)

I used an adam optimizer so that manually training the learning rate wasn't necessary.
