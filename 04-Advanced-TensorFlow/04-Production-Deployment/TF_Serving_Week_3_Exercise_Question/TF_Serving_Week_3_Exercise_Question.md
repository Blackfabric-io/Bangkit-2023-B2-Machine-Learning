TensorBoard with Fashion MNIST

*# Load the TensorBoard notebook extension.*

print("TensorFlow version: ", tf.__version__)

Load the Fashion-MNIST Dataset

per category. The images have a size of 28×28 pixels.

*T-short/top, 1 -> Trouser, etc.*

evolves over time.

%load_ext tensorboard

from os import getcwd

*# Load the data.*

import matplotlib.pyplot as plt

from tensorflow import keras from datetime import datetime

TensorFlow version: 2.9.1

import itertools import numpy as np import sklearn.metrics import tensorflow as tf

Setup

import io

In this week's exercise you will train a convolutional neural network to classify images of the Fashion MNIST dataset and you will use TensorBoard to explore how it's confusion matrix

We are going to use a CNN to classify images in the the Fashion-MNIST dataset. This dataset consist of 70,000 grayscale images of fashion products from 10 categories, with 7,000 images

First, we load the data. Even though these are really images, we will load them as NumPy arrays

and not as binary image objects. The data is already divided into training and testing sets.

train_images = np.load(f"{getcwd()}/../tmp2/train_images.npy") train_labels = np.load(f"{getcwd()}/../tmp2/train_labels.npy")

test_images = np.load(f"{getcwd()}/../tmp2/test_images.npy") test_labels = np.load(f"{getcwd()}/../tmp2/test_labels.npy")

*# The labels of the images are integers representing classes. # Here we set the Names of the integer classes, i.e., 0 ->* 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images is a NumPy array with shape (60000, 28, 28) and test_images is a NumPy array with shape (10000, 28, 28). However, our model expects arrays with shape (batch_size, height, width, channels) . Therefore, we must reshape our NumPy arrays to also include the number of color channels. Since the images are grayscale, we will set channels to 1. We will also normalize the values of our NumPy arrays to be in the range

train_images = train_images.reshape(60000, 28, 28, 1)

test_images = test_images.reshape(10000, 28, 28, 1)

tf.keras.layers.Conv2D(64, (3,3), activation='relu',

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

 tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')

loss='sparse_categorical_crossentropy',

gives you detailed knowledge of how your classifier is performing on test data.

When training a classifier, it's often useful to see the confusion matrix. The confusion matrix

In the cell below, we will define a function that returns a Matplotlib figure containing the plotted

Format the Images

*# Pre-process images*

Build the Model

*# Build the model*

train_images = train_images / 255.0

test_images = test_images / 255.0

We will build a simple CNN and compile it.

input_shape=(28, 28, 1)),

tf.keras.layers.Flatten(),

model.compile(optimizer='adam',

Plot Confusion Matrix

confusion matrix.

metrics=['accuracy'])

model = tf.keras.models.Sequential([

tf.keras.layers.MaxPooling2D(2, 2),

tf.keras.layers.MaxPooling2D(2,2),

[0,1].

])

def plot_confusion_matrix(cm, class_names):

figure = plt.figure(figsize=(8, 8))

plt.yticks(tick_marks, class_names)

*# Normalize the confusion matrix.*

threshold = cm.max() / 2.

range(cm.shape[1])):

plt.tight_layout()

return figure

!rm -rf logs/image

plt.ylabel('True label')

TensorBoard Callback

plt.xlabel('Predicted label')

*# GRADED CODE: tensorboard_callback*

*# Clear logs prior to logging data.*

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names, rotation=45)

*# Use white text if squares are dark; otherwise black.*

 color = "white" if cm[i, j] > threshold else "black" plt.text(j, i, cm[i, j], horizontalalignment="center",

the cell below, you will create a Keras TensorBoard callback to log basic metrics.

for i, j in itertools.product(range(cm.shape[0]),

plt.title("Confusion matrix")

 *Returns a matplotlib figure containing the plotted confusion* 

 *cm (array, shape = [n, n]): a confusion matrix of integer* 

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

 *class_names (array, shape = [n]): String names of the integer* 

cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],

We are now ready to train the CNN and regularly log the confusion matrix during the process. In

*"""*

*matrix.*

*classes*

*classes """*

plt.colorbar()

 *Args:*

decimals=2)

color=color)

*# UNQ_C1*

*# Create log directory*

to PNG format so it can be written.

def plot_to_image(figure):

buf = io.BytesIO()

 *# the notebook.* plt.close(figure)

buf.seek(0)

return image

*# GRADED function: plot_to_image*

plt.savefig(buf, format='png')

*# UNQ_C2*

*"""*

*this call. """*

and

*inside*

*# log files to be parsed by TensorBoard.*

Convert Matplotlib Figure to PNG

logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")

*# EXERCISE: Define a TensorBoard callback. Use the log_dir parameter # to specify the path to the directory where you want to save the*

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

Unfortunately, the Matplotlib file format cannot be logged as an image, but the PNG file format can be logged. So, you will create a helper function that takes a Matplotlib figure and converts it

 *Converts the matplotlib plot specified by 'figure' to a PNG image* 

 *returns it. The supplied figure is closed and inaccessible after* 

*# EXERCISE: Use plt.savefig to save the plot to a PNG in memory.*

*# Closing the figure prevents it from being displayed directly* 

*# EXERCISE: Use tf.image.decode_png to convert the PNG buffer*

image = tf.image.decode_png(buf.getvalue(), channels=4)

 *# EXERCISE: Use tf.expand_dims to add the batch dimension* image = tf.image.decode_png(buf.getvalue(), channels=4)

*# to a TF image. Make sure you use 4 channels.*

file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

Confusion Matrix

*# UNQ_C3*

*test_images.*

cm_callback =

code below.

In the cell below, you will define a function that calculates the confusion matrix.

*# EXERCISE: Use the model to predict the values from the* 

 *# EXERCISE: Calculate the confusion matrix using sklearn.metrics* cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)

figure = plot_confusion_matrix(cm, class_names=class_names)

tf.summary.image("Confusion Matrix", cm_image, step=epoch)

keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

The next step will be to run the code shown below to render the TensorBoard. Unfortunately, TensorBoard cannot be rendered within the Coursera environment. Therefore, we won't run the

However, you are welcome to download the notebook and run the above code locally on your

machine or in Google's Colab to see TensorBoard in action. Below are some example

*# GRADED function: log_confusion_matrix*

def log_confusion_matrix(epoch, logs):

cm_image = plot_to_image(figure)

with file_writer_cm.as_default():

*# Define the per-epoch callback.*

%tensorboard --logdir logs/image

verbose=0, *# Suppress chatty output*

screenshots that you should see when executing the code:

 callbacks=[tensorboard_callback, cm_callback], validation_data=(test_images, test_labels))

Running TensorBoard

*# Start TensorBoard.*

*# Train the classifier.* model.fit(train_images, train_labels, epochs=5,

test_pred_raw = model.predict(test_images)

test_pred = np.argmax(test_pred_raw, axis=1)

*# Log the confusion matrix as an image summary.*

Submission Instructions

*Assignment' button above.*

<!-- Save the notebook -->

window.onbeforeunload = null

IPython.notebook.save_checkpoint();

IPython.notebook.session.delete();

<IPython.core.display.Javascript object>

<!-- Shutdown and close the notebook -->

<IPython.core.display.Javascript object>

%%javascript

%%javascript

window.close();

*# Now click `File -> Save and Checkpoint` and press the 'Submit* 

please run the two cells below to save your

work and close the Notebook. This frees up

resources for your fellow learners.

When you're done or would like to take a break,

