#**Behavioral Cloning**

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json is the structure of the model
* model.h5 containing a trained convolution neural network weights
* writeup_report.md summarizing the results
* train.ipynb is a Jupyter notebook I used to evaluate preprocessing

The model can be run with:
```sh
python drive.py model.json
```
##Model Architecture and Training Strategy

###1. Solution Design Approach

To capture high-level patterns, I knew that I would need a convolutional neural network. I first tried transfer learning on VGG and Inception models, but these were slow and performed poorly. I then experimented with my own model, but found the NVIDIA model and decided to try that.

The model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 86-98)

The model then uses a number of densely connected layers with RELU activations to introduce non-linearities. These are combined into a single output, which represents the steering angle.

###2. Creation of the Training Set & Training Process

I drove the car around the track a few times, then tried to record recovery from the sides. I tried removing some of the zero-angle data points, as this biases the car towards no steering, which can lead it to going off the track.

I compiled my model to use the Mean Squared Error loss functions and to use the Adam optimizer, so to not need to change the learning rate.

###3. Evaluation

