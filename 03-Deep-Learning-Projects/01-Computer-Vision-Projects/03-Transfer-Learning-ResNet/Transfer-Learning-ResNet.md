Welcome to this assignment! This week, you are going to use a technique called Transfer Learning in which you utilize an already trained

*NOTE: To prevent errors from the autograder, please avoid editing or deleting non-graded cells in this notebook . Please only put your solutions in*

*between the* ### *START CODE HERE and* ### END *CODE HERE code comments, and refrain from adding any new cells.*

For this assignment, you will use the Horse or Human dataset , which contains images of horses and humans.

!wget -q -P /content/ https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip

!wget -q -P /content/ https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip

This dataset already has an structure that is compatible with Keras' flow_from_directory so you don't need to move the images into

subdirectories as you did in the previous assignments. However, it is still a good idea to save the paths of the images so you can use them later

network to help you solve a similar problem to the one it was originally trained to solve.

Let's get started!

# grader-required-cell

import matplotlib.pyplot as plt import tensorflow as tf

from tensorflow.keras import layers from tensorflow.keras import Model

from tensorflow.keras.optimizers import RMSprop

# Get the Horse or Human training dataset

# Get the Horse or Human validation dataset

val_local_zip = './validation-horse-or-human.zip' zip_ref = zipfile.ZipFile(val_local_zip, 'r') zip_ref.extractall('/tmp/validation')

# Define the training and validation base directories

train_horses_dir = os.path.join(train_dir, 'horses')

train_humans_dir = os.path.join(train_dir, 'humans')

# Check the number of images for each class and set

There are 500 images of horses for training. There are 527 images of humans for training.

validation_horses_dir = os.path.join(validation_dir, 'horses')

validation_humans_dir = os.path.join(validation_dir, 'humans')

print(f"There are {len(os.listdir(train_horses_dir))} images of horses for training.\n") print(f"There are {len(os.listdir(train_humans_dir))} images of humans for training.\n") print(f"There are {len(os.listdir(validation_horses_dir))} images of horses for validation.\n") print(f"There are {len(os.listdir(validation_humans_dir))} images of humans for validation.\n")

test_local_zip = './horse-or-human.zip' zip_ref = zipfile.ZipFile(test_local_zip, 'r')

zip_ref.extractall('/tmp/training')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

Download the training and validation sets by running the cell below:

from tensorflow.keras.utils import img_to_array, load_img

import os import zipfile

Dataset

zip_ref.close()

# grader-required-cell

train_dir = '/tmp/training' validation_dir = '/tmp/validation'

# Directory with training horse pictures

# Directory with training humans pictures

# Directory with validation horse pictures

# Directory with validation human pictures

on:

Week 3: Transfer Learning

There are 128 images of horses for validation. There are 128 images of humans for validation.

Now take a look at a sample image of each one of the classes:

plt.imshow(load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}"))

plt.imshow(load_img(f"{os.path.join(train_humans_dir, os.listdir(train_humans_dir)[0])}"))

matplotlib makes it easy to see that these images have a resolution of 300x300 and are colored, but you can double check this by using the

sample_image = load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}")

Sample horse image:

print("\nSample human image:")

# grader-required-cell

plt.show()

plt.show()

print("Sample horse image:")

Sample human image:

code below:

# grader-required-cell

# Load the first example of a horse

sample_array = img_to_array(sample_image)

# Convert the image into its numpy array representation

print(f"Each image has shape: {sample_array.shape}")

Each image has shape: (300, 300, 3)

this, complete the train_val_generators function below:

def train_val_generators(TRAINING_DIR, VALIDATION_DIR):

Creates the training and validation data generators

# Instantiate the ImageDataGenerator class

train_datagen = ImageDataGenerator(rescale = 1./255.,

# Remember that validation data should not be augmented validation_datagen = ImageDataGenerator(rescale = 1./255.)

return train_generator, validation_generator

Found 1027 images belonging to 2 classes. Found 256 images belonging to 2 classes.

Transfer learning - Create the pre-trained model

Found 1027 images belonging to 2 classes. Found 256 images belonging to 2 classes.

Training and Validation Generators

yielding much quicker training times without compromising the accuracy!

TRAINING_DIR (string): directory path containing the training images

train_generator, validation_generator: tuple containing the generators

# Pass in the appropriate arguments to the flow_from_directory method train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,

# Pass in the appropriate arguments to the flow_from_directory method

# Don't forget to normalize pixel values and set arguments to augment the images

rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)

validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,

train_generator, validation_generator = train_val_generators(train_dir, validation_dir)

VALIDATION_DIR (string): directory path containing the testing/validation images

color.

# grader-required-cell

"""

Args:

Returns:

### START CODE HERE

### END CODE HERE

# grader-required-cell # Test your generators

**Expected Output:**

"""

# GRADED FUNCTION: train_val_generators

As expected, the sample image has a resolution of 300x300 and the last dimension is used for each one of the RGB channels to represent

Now that you know the images you are dealing with, it is time for you to code the generators that will fed these images to your Network. For

**Important Note:** The images have a resolution of 300x300 but the flow_from_directory method you will use allows you to set a target resolution. In this case, **set a target_size of (150, 150)**. This will heavily lower the number of trainable parameters in your final networ

> batch_size=32, class_mode='binary', target_size=(150, 150))

> > batch_size=32, class_mode='binary', target_size=(150, 150))

Download the inception V3 weights into the /tmp/ directory:

HTTP request sent, awaiting response... 200 OK Length: 87910968 (84M) [application/x-hdf]

-O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

Saving to: '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' /tmp/inception_v3_w 100%[===================>] 83.84M 21.7MB/s in 4.9s

Now load the InceptionV3 model and save the path to the weights you just downloaded:

# Create an instance of the inception model from the local pre-trained weights local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

local_weights_file (string): path pointing to a pretrained weights H5 file

include_top = False, weights = None)

Check that everything went well by comparing the last few rows of the model summary to the expected output:

__________________________________________________________________________________________________

==================================================================================================

Layer (type) Output Shape Param # Connected to

conv2d (Conv2D) (None, 74, 74, 32) 864 ['input_1[0][0]']

input_1 (InputLayer) [(None, 150, 150, 3)] 0 []

pre_trained_model: the initialized InceptionV3 model

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:

layer.trainable = False

return pre_trained_model

### END CODE HERE

# grader-required-cell

# Print the model summary pre_trained_model.summary() Model: "inception_v3"

pre_trained_model = InceptionV3(input_shape = (150, 150, 3),

# Make all the layers in the pre-trained model non-trainable

pre_trained_model = create_pre_trained_model(local_weights_file)

new resolution for the images instead of the native 300x300) and make all of the layers non-trainable:

from tensorflow.keras.applications.inception_v3 import InceptionV3

https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \

Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.200.207|:443... connected.

Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.200.207, 74.125.130.207, 74.125.68.207, ...

--2023-11-18 04:15:22-- https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

2023-11-18 04:15:28 (17.0 MB/s) - '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' saved [87910968/87910968]

Complete the create_pre_trained_model function below. You should specify the correct input_shape for the model (remember that you set a

# Download the inception v3 weights !wget --no-check-certificate \

# grader-required-cell

# grader-required-cell

### START CODE HERE

"""

Args:

"""

Returns:

# GRADED FUNCTION: create_pre_trained_model def create_pre_trained_model(local_weights_file):

Initializes an InceptionV3 model.

# Import the inception model

batch_normalization (Batch (None, 74, 74, 32) 96 ['conv2d[0][0]']

 conv2d_1 (Conv2D) (None, 72, 72, 32) 9216 ['activation[0][0]'] batch_normalization_1 (Bat (None, 72, 72, 32) 96 ['conv2d_1[0][0]']

 conv2d_2 (Conv2D) (None, 72, 72, 64) 18432 ['activation_1[0][0]'] batch_normalization_2 (Bat (None, 72, 72, 64) 192 ['conv2d_2[0][0]']

max_pooling2d (MaxPooling2 (None, 35, 35, 64) 0 ['activation_2[0][0]']

 conv2d_3 (Conv2D) (None, 35, 35, 80) 5120 ['max_pooling2d[0][0]'] batch_normalization_3 (Bat (None, 35, 35, 80) 240 ['conv2d_3[0][0]']

 conv2d_4 (Conv2D) (None, 33, 33, 192) 138240 ['activation_3[0][0]'] batch_normalization_4 (Bat (None, 33, 33, 192) 576 ['conv2d_4[0][0]']

max_pooling2d_1 (MaxPoolin (None, 16, 16, 192) 0 ['activation_4[0][0]']

batch_normalization_8 (Bat (None, 16, 16, 64) 192 ['conv2d_8[0][0]']

__________________________________________________________________________________________________ activation_273 (Activation) (None, 3, 3, 320) 0 batch_normalization_v1_273[0][0] __________________________________________________________________________________________________

__________________________________________________________________________________________________

__________________________________________________________________________________________________ activation_281 (Activation) (None, 3, 3, 192) 0 batch_normalization_v1_281[0][0] __________________________________________________________________________________________________

==================================================================================================

To check that all the layers in the model were set to be non-trainable, you can also run the cell below:

conv2d_8 (Conv2D) (None, 16, 16, 64) 12288 ['max_pooling2d_1[0][0]']

activation_8 (Activation) (None, 16, 16, 64) 0 ['batch_normalization_8[0][0]'

activation_276[0][0]

activation_280[0][0]

mixed9_1[0][0] concatenate_5[0][0] activation_281[0][0]

]

]

]

]

batch_normalization_v1_281 (Bat (None, 3, 3, 192) 576 conv2d_281[0][0]

mixed9_1 (Concatenate) (None, 3, 3, 768) 0 activation_275[0][0]

concatenate_5 (Concatenate) (None, 3, 3, 768) 0 activation_279[0][0]

mixed10 (Concatenate) (None, 3, 3, 2048) 0 activation_273[0][0]

activation (Activation) (None, 74, 74, 32) 0 ['batch_normalization[0][0]']

activation_1 (Activation) (None, 72, 72, 32) 0 ['batch_normalization_1[0][0]'

activation_2 (Activation) (None, 72, 72, 64) 0 ['batch_normalization_2[0][0]'

activation_3 (Activation) (None, 35, 35, 80) 0 ['batch_normalization_3[0][0]'

activation_4 (Activation) (None, 33, 33, 192) 0 ['batch_normalization_4[0][0]'

Normalization)

chNormalization)

chNormalization)

chNormalization)

chNormalization)

chNormalization)

g2D)

**Expected Output:**

Total params: 21,802,784 Trainable params: 0

# grader-required-cell

Non-trainable params: 21,802,784

total_params = pre_trained_model.count_params()

D)

num_trainable_params = sum([w.shape.num_elements() for w in pre_trained_model.trainable_weights])

You have already worked with callbacks in the first course of this specialization so the callback to stop training once an accuracy of 99.9%

For this you will need the last output of the pre-trained model, since this will be the input for your own. Complete the output_of_last_layer

**Note:** For grading purposes use the mixed7 layer as the last layer of the pre-trained model. However, after submitting feel free to come back

last layer output: KerasTensor(type_spec=TensorSpec(shape=(None, 7, 7, 768), dtype=tf.float32, name=None), name='mixed7/concat:0',

print(f"There are {total_params:,} total parameters in this model.")

# Define a Callback class that stops training once accuracy reaches 99.9%

Now that the pre-trained model is ready, you need to "glue" it to your own model to solve the task at hand.

pre_trained_model (tf.keras Model): model to get the last layer output from

print("\nReached 99.9% accuracy so cancelling training!")

Pipelining the pre-trained model with your own

There are 21,802,784 total parameters in this model. There are 0 trainable parameters in this model.

There are 21,802,784 total parameters in this model. There are 0 trainable parameters in this model.

**Expected Output:**

reached, is provided for you:

# grader-required-cell

function below.

"""

Args:

"""

Returns:

here and play around with this.

# GRADED FUNCTION: output_of_last_layer def output_of_last_layer(pre_trained_model):

Gets the last layer output of a model

last_output = last_desired_layer.output print('last layer output: ', last_output)

Check that everything works as expected:

**Expected Output (if mixed7 layer was used):**

last_output: output of the model's last layer

last_output = output_of_last_layer(pre_trained_model) last layer output shape: (None, 7, 7, 768)

last_desired_layer = pre_trained_model.get_layer('mixed7') print('last layer output shape: ', last_desired_layer.output_shape)

# grader-required-cell

### START CODE HERE

### END CODE HERE return last_output

# grader-required-cell

Creating callbacks for later

class myCallback(tf.keras.callbacks.Callback): def on_epoch_end(self, epoch, logs={}): if(logs.get('accuracy')>0.999):

self.model.stop_training = True

print(f"There are {num_trainable_params:,} trainable parameters in this model.")

last layer output shape: (None, 7, 7, 768)

# Print the type of the pre-trained model

directly as output when creating the final mode

def create_final_model(pre_trained_model, last_output):

Appends a custom model to a pre-trained model

# Flatten the output layer to 1 dimension x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation='relu')(x)

# Add a final sigmoid layer for classification x = layers.Dense (1, activation='sigmoid')(x)

# Create the complete model by using the Model class model = Model(inputs=pre_trained_model.input, outputs=x)

model.compile(optimizer = RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

model = create_final_model(pre_trained_model, last_output)

print(f"There are {total_params:,} total parameters in this model.")

There are 47,512,481 total parameters in this model. There are 38,537,217 trainable parameters in this model.

num_trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])

print(f"There are {num_trainable_params:,} trainable parameters in this model.")

# GRADED FUNCTION: create_final_model

model: the combined model

# Add a dropout rate of 0.2 x = layers.Dropout(0.2)(x)

### START CODE HERE

# Compile the model

### END CODE HERE

# grader-required-cell

# Inspect parameters

# Save your model in a variable

total_params = model.count_params()

return model

print(f"The pretrained model has type: {type(pre_trained_model)}")

been created using it. Let's double check this firs

# grader-required-cell

instantiate a Model in the docs.

# grader-required-cell

"""

Args:

Returns:

"""

last layer output: KerasTensor(type_spec=TensorSpec(shape=(None, 7, 7, 768), dtype=tf.float32, name=None), name='mixed7/concat:0', description="c

Complete the create_final_model function below. You will need to use Tensorflow's Functional API for this since the pretrained model h

To create the final model, you will use Keras' Model class by defining the appropriate inputs and outputs as described in the first

Note that you can get the input from any existing model by using its input attribute and by using the Funcional API you can use the last layer

Now you will create the final model by adding some additional layers on top of the pre-trained mode

The pretrained model has type: <class 'keras.src.engine.functional.Functional'>

pre_trained_model (tf.keras Model): model that will accept the train/test inputs

last_output (tensor): last layer output of the pre-trained model

# Add a fully connected layer with 1024 hidden units and ReLU activation

**Expected Output:**

Now train the model:

Epoch 1/100

Epoch 2/100

cool!

Wow, that is a lot of parameters!

# (It should take a few epochs) callbacks = myCallback()

history = model.fit(train_generator,

acc = history.history['accuracy'] val_acc = history.history['val_accuracy']

loss = history.history['loss'] val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.legend(loc=0) plt.figure() plt.show()

many more parameters are for that case.

# fires, and stops training at 99.9% accuracy

epochs = 100, verbose = 2, callbacks=callbacks)

Reached 99.9% accuracy so cancelling training!

# Plot the training and validation accuracies for each epoch

plt.plot(epochs, acc, 'r', label='Training accuracy') plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

<Figure size 640x480 with 0 Axes>

There are 47,512,481 total parameters in this model. There are 38,537,217 trainable parameters in this model.

# Run this and see how many epochs it should take before the callback

validation_data = validation_generator,

Now take a quick look at the training and validation accuracies for each epoch of training:

After submitting your assignment later, try re-running this notebook but use the original resolution of 300x300, you will be surprised to see how

33/33 - 13s - loss: 0.0042 - accuracy: 0.9981 - val_loss: 0.0239 - val_accuracy: 0.9883 - 13s/epoch - 387ms/step

33/33 - 13s - loss: 0.0061 - accuracy: 0.9990 - val_loss: 3.1634e-04 - val_accuracy: 1.0000 - 13s/epoch - 389ms/step

The training should have stopped after less than 10 epochs and it should have reached an accuracy over 99,9% (firing the callback). Th happened so quickly because of the pre-trained model you used, which already contained information to classify humans from horses. Really You will need to submit your solution notebook for grading. The following code cells will check if this notebook's grader metadata (i.e. hidden data in the notebook needed for grading) is not modified by your workspace. This will ensure that the autograder can evaluate your co

*if the metadata is missing*: A new notebook with your solutions will be created on this Colab workspace. It should be downloaded

-> Download .ipynb. You can name it anything you want as long as it is a valid .ipynb (jupyter notebook) fil

--2023-11-18 04:26:04-- https://storage.googleapis.com/tensorflow-1-public/colab_metadata_checker.py

Connecting to storage.googleapis.com (storage.googleapis.com)|142.250.4.207|:443... connected.

Resolving storage.googleapis.com (storage.googleapis.com)... 142.250.4.207, 142.251.10.207, 142.251.12.207, ...

*if the metadata is intact*: Download the current notebook. Click on the File tab on the upper left corner of the screen then click on Download

Grader metadata detected! You can download this notebook by clicking `File > Download > Download as .ipynb` and submit it to the gra

*Click the Folder Refresh icon in the File Manager to see the latest files in the workspace. You should see a file ending with a _fixed.ipy*

You have successfully implemented a convolutional neural network that leverages a pre-trained network to help you solve the problem of

properly. Depending on its output, you will either:

# Download metadata checker

import colab_metadata_checker

Download your notebook for grading

automatically and you can submit that to the grader.

HTTP request sent, awaiting response... 200 OK Length: 1997 (2.0K) [text/x-python-script] Saving to: 'colab_metadata_checker.py'

colab_metadata_checker.run('C2W3_Assignment_fixed.ipynb')

**Please disregard the following note if the notebook metadata is detected**

*Right-click on that file to save locally and submit it to the grade*

**Congratulations on finishing this week's assignmen**

classifying humans from horses.

**Keep it up!**

!wget -nc https://storage.googleapis.com/tensorflow-1-public/colab_metadata_checker.py

colab_metadata_chec 100%[===================>] 1.95K --.-KB/s in 0s 2023-11-18 04:26:04 (40.9 MB/s) - 'colab_metadata_checker.py' saved [1997/1997]

# Please see the output of this cell to see which file you need to submit to the grader

*Note: Just in case the download fails for the second point above, you can also do these steps: Click the Folder icon on the left side of this screen to open the File Manager.*

