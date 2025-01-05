Exporting an MNIST Classifier in SavedModel

In this exercise, we will learn on how to create models for TensorFlow Hub. You will be tasked

We will start by creating a class called MNIST. This class will load the MNIST dataset, preprocess the images from the dataset, and build a CNN based classifier. This class will also have some

In the cell below, fill in the missing code and create the following Keras Sequential model:

 _________________________________________________________________ Layer (type) Output Shape Param # =================================================================

_________________________________________________________________

_________________________________________________________________

_________________________________________________________________

_________________________________________________________________

_________________________________________________________________

_________________________________________________________________

_________________________________________________________________

lambda (Lambda) (None, 28, 28, 1) 0

conv2d (Conv2D) (None, 28, 28, 8) 80

max_pooling2d (MaxPooling2D) (None, 14, 14, 8) 0

max_pooling2d_1 (MaxPooling2 (None, 7, 7, 16) 0

flatten (Flatten) (None, 1568) 0

conv2d_1 (Conv2D) (None, 14, 14, 16) 1168

conv2d_2 (Conv2D) (None, 7, 7, 32) 4640

• Creating a simple MNIST classifier and evaluating its accuracy.

• Importing this TF Hub Module to be used with Keras Layers.

Format

with performing the following tasks:

• Exporting it into SavedModel.

import tensorflow_hub as hub

import numpy as np

import tensorflow as tf

from os import getcwd from absl import logging

• Hosting the model as TF Hub Module.

import tensorflow_datasets as tfds

logging.set_verbosity(logging.ERROR)

Create an MNIST Classifier

methods to train, test, and save our model.

Model: "sequential"

 dense (Dense) (None, 128) 200832 _________________________________________________________________

=================================================================

Notice that we are using a tf.keras.layers.Lambda layer at the beginning of our model.

The Lambda layer exists so that arbitrary TensorFlow functions can be used when constructing Sequential and Functional API models. Lambda layers are best suited for simple operations.

def __init__(self, export_path, buffer_size=1000, batch_size=32,

 *# EXERCISE: Cast x to tf.float32 using the tf.cast() function. # You should also normalize the values of x to be in the range*

*# EXERCISE: Build the model according to the model summary* 

tf.keras.layers.Input(shape=(28, 28, 1), dtype=tf.uint8),

*# Create a Conv2D layer with 8 filters, a kernel size of 3*

*# Use a Lambda layer to use the self.preprocess_fn* 

dense_1 (Dense) (None, 10) 1290

Lambda layers are used to wrap arbitrary expressions as a Layer object:

learning_rate=1e-3, epochs=10):

 self._export_path = export_path self._buffer_size = buffer_size self._batch_size = batch_size

self._epochs = epochs

def preprocess_fn(self, x):

self._build_model()

def _build_model(self):

self._prepare_dataset()

self._learning_rate = learning_rate

*# Function to preprocess the images.*

x = tf.cast(x, tf.float32) / 255.0

self._model = tf.keras.models.Sequential([

 *# defined above to preprocess the images.* tf.keras.layers.Lambda(self.preprocess_fn),

self.train_dataset, self.test_dataset =

tf.keras.layers.Lambda(expression)

class MNIST:

*[0, 1].*

return x

*shown above.*

*function*

*# and padding='same'.*

*# and padding='same'.*

*# and padding='same'.*

 *# model summary shown above.* tf.keras.layers.Flatten(), tf.keras.layers.Dense(128),

 *# Set the metrics to accuracy.* metrics_list = ['accuracy']

filePath = f"{getcwd()}/../tmp2"

*# Compile the model.*

def _prepare_dataset(self):

metrics=metrics_list)

*# learning rate to self._learning_rate.*

loss_fn = "sparse_categorical_crossentropy"

self._model.compile(optimizer_fn, loss=loss_fn,

3

3

*sure to use*

])

self._learning_rate)

tf.keras.layers.Conv2D(8, (3, 3), padding='same'),

*# Create a MaxPool2D() layer. Use default values.*

tf.keras.layers.Conv2D(16, (3, 3), padding='same'),

tf.keras.layers.Conv2D(32, (3, 3), padding='same'),

tf.keras.layers.Dense(10, activation='softmax')

*# Create a MaxPool2D() layer. Use default values.*

*# Create a Conv2D layer with 16 filters, a kernel size of* 

*# Create a Conv2D layer with 32 filters, a kernel size of* 

*# Create the Flatten and Dense layers as described in the* 

*# EXERCISE: Define the optimizer, loss function and metrics.*

*# Use sparse_categorical_crossentropy as your loss function.*

*# EXERCISE: Load the MNIST dataset using tfds.load(). Make* 

*# Use the tf.keras.optimizers.Adam optimizer and set the*

optimizer_fn = tf.keras.optimizers.Adam(learning_rate =

tf.keras.layers.MaxPooling2D((2, 2)),

tf.keras.layers.MaxPooling2D((2, 2)),

*# the argument data_dir=filePath. You should load the images* 

*# as their corresponding labels and load both the test and* 

*# EXERCISE: Extract the 'train' and 'test' splits from the* 

*# EXERCISE: Shuffle and batch the self.train_dataset. Use* 

*# as the shuffling buffer and self._batch_size as the batch* 

self.train_dataset.shuffle(self._buffer_size).batch(self._batch_size)

*# EXERCISE: Batch the self.test_dataset. Use a batch size of* 

*# Print the metric values on which the model is being* 

tf.saved_model.save(self._model, self._export_path)

for name, value in zip(self._model.metrics_names, results):

We will now use the MNIST class we created above to create an mnist object. When creating our mnist object we will use a dictionary to pass our training parameters. We will then call the train and export_model methods to train and save our model, respectively. Finally, we call

 *# Train the model for specified number of epochs.* self._model.fit(dataset_tr, epochs=self._epochs)

dataset_te = self.test_dataset.batch(32)

results = self._model.evaluate(dataset_te)

print("%s: %.3f" % (name, value))

Train, Evaluate, and Save the Model

the test method to evaluate our model after training.

dataset = tfds.load('mnist', data_dir=filePath,

split=['train', 'test'], as_supervised=True)

train_dataset, test_dataset = dataset

return train_dataset, test_dataset

*as well*

*evaluated on.*

32.

*train splits.*

*dataset above.*

def train(self):

*self._buffer_size*

*size for batching.*  dataset_tr =

def test(self):

*# Evaluate the dataset*

 def export_model(self): *# Save the model.*

**NOTE:** It will take about 12 minutes to train the model for 5 epochs.

*# UNQ_C1*

}

*# GRADED CODE: MNIST*

'epochs': 5

mnist = MNIST(**args)

*# Train the model.*

*# Save the model.* mnist.export_model()

mnist.train()

mnist.test()

Epoch 1/5

Epoch 2/5

Epoch 3/5

Epoch 4/5

Epoch 5/5

*# Define the training parameters.*

 'buffer_size': 1000, 'batch_size': 32,

*# Create the mnist object.* 

0.4349 - accuracy: 0.8467

0.3318 - accuracy: 0.8830

0.3068 - accuracy: 0.8910

0.2905 - accuracy: 0.8965

0.2796 - accuracy: 0.8985

- accuracy: 0.8827

loss: 0.339 accuracy: 0.883

*# Evaluate the trained MNIST model.*

1875/1875 [==============================] - 39s 17ms/step - loss:

1875/1875 [==============================] - 12s 7ms/step - loss:

1875/1875 [==============================] - 13s 7ms/step - loss:

1875/1875 [==============================] - 12s 7ms/step - loss:

1875/1875 [==============================] - 13s 7ms/step - loss:

313/313 [==============================] - 5s 15ms/step - loss: 0.3386

INFO:tensorflow:Assets written to: ./saved_model/assets

INFO:tensorflow:Assets written to: ./saved_model/assets

'learning_rate': 1e-3,

args = {'export_path': './saved_model',

Create a Tarball

Inspect the Tarball

*# Inspect the tarball.* !tar -tf module.tar.gz

./variables/variables.index

Simulate Server Conditions

directory where our SavedModel is stored.

./variables/variables.index

!tar xvzf module.tar.gz -C ./module

./variables/variables.data-00001-of-00002 ./variables/variables.data-00000-of-00001 ./variables/variables.data-00000-of-00002

*# Create a tarball from the SavedModel.*

./variables/variables.data-00001-of-00002 ./variables/variables.data-00000-of-00001 ./variables/variables.data-00000-of-00002

!tar -cz -f module.tar.gz -C ./saved_model .

SavedModel.

SavedModel.

./variables/

./assets/

./saved_model.pb

!rm -rf ./module !mkdir -p module

./saved_model.pb

./variables/

./

./

The export_model method saved our model in the TensorFlow SavedModel format in the ./saved_model directory. The SavedModel format saves our model and its weights in various files and directories. This makes it difficult to distribute our model. Therefore, it is convenient to create a single compressed file that contains all the files and folders of our model. To do this, we will use the tar archiving program to create a tarball (similar to a Zip file) that contains our

We can uncompress our tarball to make sure it has all the files and folders from our

Once we have verified our tarball, we can now simulate server conditions. In a normal scenario, we will fetch our TF Hub module from a remote server using the module's handle. However, since this notebook cannot host the server, we will instead point the module handle to the

tar: ./variables: Cannot change ownership to uid 65534, gid 65534:

Operation not permitted

*# Define the module handle.* MODULE_HANDLE = './module'

Load the TF Hub Module

model = hub.load(MODULE_HANDLE)

*# GRADED CODE: dataset, test_dataset*

filePath = f"{getcwd()}/../tmp2"

filePath, as_supervised=True)

test_dataset = dataset.batch(32)

outputs = model(batch_data[0])

for batch_data in test_dataset.take(1):

 outputs = np.argmax(outputs, axis=-1) print('Predicted Labels:', outputs)

Test the TF Hub Module

*# GRADED CODE: model*

tar: .: Cannot change ownership to uid 65534, gid 65534: Operation not

tar: Exiting with failure status due to previous errors

*# EXERCISE: Load the TF Hub module using the hub.load API.*

*# EXERCISE: Load the MNIST 'test' split using tfds.load(). # Make sure to use the argument data_dir=filePath. You*

*# EXERCISE: Batch the dataset using a batch size of 32.*

*# Test the TF Hub module for a single batch of data*

print('True Labels: ', batch_data[1].numpy())

*# should load the images along with their corresponding labels.*

dataset = tfds.load('mnist', split=tfds.Split.TEST, data_dir =

Predicted Labels: [4 4 9 7 5 1 0 5 7 4 0 8 2 3 9 6 5 7 2 2 0 4 4 4 0 7

True Labels: [4 4 9 7 5 1 0 5 7 4 0 8 2 3 9 0 7 7 2 2 0 4 4 4 2 7

We can see that the model correctly predicts the labels for most images in the batch.

We will now test our TF Hub module with images from the test split of the MNIST dataset.

./assets/

permitted

*# UNQ_C2*

*# UNQ_C3*

7 4 2 4 7 5]

7 4 2 4 7 5]

Evaluate the Model Using Keras

*# GRADED CODE: dataset, test_dataset*

input_shape=[28,28,1], dtype=tf.uint8)])

*# Evaluate the model on the test_dataset.* results = model.evaluate(test_dataset)

*# the dtype parameter.*

*# Compile the model.*

- accuracy: 0.8827

loss: 0.339 accuracy: 0.883

model.compile(optimizer='adam',

metrics=['accuracy'])

print("%s: %.3f" % (name, value))

Submission Instructions

*Assignment' button above.*

*# UNQ_C4*

In the cell below, you will integrate the TensorFlow Hub module into the high level Keras API.

*# should make sure to use the correct values for the output_shape, # and input_shape parameters. You should also use tf.uint8 for*

model = tf.keras.Sequential([hub.KerasLayer(model, output_shape=[10],

313/313 [==============================] - 5s 15ms/step - loss: 0.3386

*# Print the metric values on which the model is being evaluated on.*

*# Now click `File -> Save and Checkpoint` and press the 'Submit* 

*# EXERCISE: Integrate the TensorFlow Hub module into a Keras # sequential model. You should use a hub.KerasLayer and you* 

loss='sparse_categorical_crossentropy',

for name, value in zip(model.metrics_names, results):

When you're done or would like to take a break,

please run the two cells below to save your

work and close the Notebook. This frees up

resources for your fellow learners.

%%javascript

%%javascript

window.close();

<!-- Save the notebook -->

window.onbeforeunload = null

IPython.notebook.save_checkpoint();

IPython.notebook.session.delete();

<IPython.core.display.Javascript object>

<!-- Shutdown and close the notebook -->

<IPython.core.display.Javascript object>

