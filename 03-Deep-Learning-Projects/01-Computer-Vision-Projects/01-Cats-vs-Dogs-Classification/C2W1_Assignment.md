Welcome to the 1st assignment of the course! This week, you will be using the famous Cats vs Dogs dataset to train a model that can classify images of dogs from images of cats. For this, you will create your own Convolutional Neural Network in Tensorflow and leverage Keras' ima

Week 1: Using CNN's with the Cats vs Dogs Dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator

Download the dataset from its original source by running the cell below.

Note that the zip file that contains the images is unzipped under the /tmp director

# Note: This is a very large dataset and will take some time to download

HTTP request sent, awaiting response... 200 OK Length: 824887076 (787M) [application/octet-stream]

Saving to: '/tmp/cats-and-dogs.zip'

source_path_dogs = os.path.join(source_path, 'Dog') source_path_cats = os.path.join(source_path, 'Cat')

> There are 12500 images of dogs. There are 12500 images of cats.

!find /tmp/PetImages/ -type f ! -name "*.jpg" -exec rm {} +

# os.listdir returns a list containing all files under the given path print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.") print(f"There are {len(os.listdir(source_path_cats))} images of cats.")

# And right click on the 'Download Manually' link to get a new URL to the dataset

/tmp/cats-and-dogs. 100%[===================>] 786.67M 59.7MB/s in 13s

# Deletes all non-image files (there are two .db files bundled into the dataset)

2023-11-14 06:40:46 (58.6 MB/s) - '/tmp/cats-and-dogs.zip' saved [824887076/824887076]

You will also create some helper functions to move the images around the filesystem so if you are not familiar with the os module be sure

*NOTE: To prevent errors from the autograder, please avoid editing or deleting non-graded cells in this notebook . Please only put your solutions in*

*between the* ### *START CODE HERE and* ### END *CODE HERE code comments, and refrain from adding any new cells.*

# If the URL doesn't work, visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

Connecting to download.microsoft.com (download.microsoft.com)|23.213.36.252|:443... connected.

"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" \

Now the images are stored within the /tmp/PetImages directory. There is a subdirectory for each class, so one for dogs and one for cats.

--2023-11-14 06:40:32-- https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.z Resolving download.microsoft.com (download.microsoft.com)... 23.213.36.252, 2600:1409:9800:480::317f, 2600:1409:9800:48e::317f

preprocessing utilities.

take a look a the docs. Let's get started!

# grader-required-cell

import tensorflow as tf

from shutil import copyfile import matplotlib.pyplot as plt

!wget --no-check-certificate \

zip_ref.extractall('/tmp')

# grader-required-cell

source_path = '/tmp/PetImages'

zip_ref.close()

-O "/tmp/cats-and-dogs.zip" local_zip = '/tmp/cats-and-dogs.zip' zip_ref = zipfile.ZipFile(local_zip, 'r')

import os import zipfile import random import shutil

**Expected Output:**

There are 12500 images of dogs. There are 12500 images of cats.

# grader-required-cell # Define root directory root_dir = '/tmp/cats-v-dogs'

"""

Args:

Returns: None """

# HINT:

try:

### START CODE HERE

### END CODE HERE

except FileExistsError:

# grader-required-cell

for subdir in dirs:

/tmp/cats-v-dogs/training /tmp/cats-v-dogs/validation /tmp/cats-v-dogs/training/cats /tmp/cats-v-dogs/training/dogs /tmp/cats-v-dogs/validation/cats /tmp/cats-v-dogs/validation/dogs

/tmp/cats-v-dogs/training /tmp/cats-v-dogs/testing /tmp/cats-v-dogs/training/dogs /tmp/cats-v-dogs/training/cats /tmp/cats-v-dogs/testing/dogs /tmp/cats-v-dogs/testing/cats

if os.path.exists(root_dir): shutil.rmtree(root_dir)

# GRADED FUNCTION: create_train_val_dirs def create_train_val_dirs(root_path):

Creates directories for the train and test sets

os.makedirs(os.path.join(root_path, 'training'))

os.makedirs(os.path.join(root_path, 'testing'))

create_train_val_dirs(root_path=root_dir)

# Test your create_train_val_dirs function

**Expected Output (directory order might vary):**

SOURCE_DIR: directory containing the fil

for rootdir, dirs, files in os.walk(root_dir):

print(os.path.join(rootdir, subdir))

Code the split_data function which takes in the following arguments:

os.makedirs(os.path.join(f'{root_path}/training', 'dogs')) os.makedirs(os.path.join(f'{root_path}/training', 'cats'))

os.makedirs(os.path.join(f'{root_path}/testing', 'dogs')) os.makedirs(os.path.join(f'{root_path}/testing', 'cats'))

'dogs'. To accomplish this, complete the create_train_val_dirs below:

# Empty directory to prevent FileExistsError is the function is run several times

root_path (string) - the base directory path to create subdirectories from

# Use os.makedirs to create your directories with intermediate subdirectories

print("You should not be seeing this since the upper directory is removed beforehand")

# Don't hardcode the paths. Use os.path.join to append the new directories to the root_path parameter

You will need a directory for cats-v-dogs, and subdirectories for training and validation. These in turn will need subdirectories for 'cats' and

TRAINING_DIR: directory that a portion of the files will be copied to (will be used for trainin VALIDATION_DIR: directory that a portion of the files will be copied to (will be used for validatio

TRAINING_DIR directory and 10% of the images will be copied to the VALIDATION_DIR directory.

The files should be randomized, so that the training set is a random sample of the files, and the validation set is made up of the remaining

All images should be checked before the copy, so if they have a zero file length, they will be omitted from the copying process. If this is the ca then your function should print out a message such as "filename is zero length, so ignoring." . **You should perform this check before the**

For example, if SOURCE_DIR is PetImages/Cat , and SPLIT_SIZE is .9 then 90% of the images in PetImages/Cat will be copied to the

SPLIT_SIZE: determines the portion of images used for training.

**split so that only non-zero images are considered when doing the actual split.**

os.path.getsize(PATH) returns the size of the fi

random.sample(list, len(list)) shuffles a l

# Find total number of files in training dir

item_source = os.path.join(SOURCE, item) if os.path.getsize(item_source) == 0:

training_number = int(len(shuffled_source) * SPLIT_SIZE)

print(f'{item} is zero length, so ignoring.')

copyfile(item_source, os.path.join(target, item))

os.listdir(DIRECTORY) returns a list with the contents of that directory.

copyfile(source, destination) copies a file from source to destinati

def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):

shuffled_source = random.sample(os.listdir(SOURCE), len(os.listdir(SOURCE)))

Hints:

# grader-required-cell

### START CODE HERE # Shuffle list

target = TRAINING

for item in shuffled_source:

# Switch copy target to TESTING if i == training_number: target = TESTING ### END CODE HERE

i = 0

else:

i += 1

# grader-required-cell

# Define paths

# Test your split_data function

CAT_SOURCE_DIR = "/tmp/PetImages/Cat/" DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"

TRAINING_DIR = "/tmp/cats-v-dogs/training/" VALIDATION_DIR = "/tmp/cats-v-dogs/validation/"

if len(os.listdir(TRAINING_CATS_DIR)) > 0: for file in os.scandir(TRAINING_CATS_DIR):

if len(os.listdir(TRAINING_DOGS_DIR)) > 0: for file in os.scandir(TRAINING_DOGS_DIR):

if len(os.listdir(VALIDATION_CATS_DIR)) > 0: for file in os.scandir(VALIDATION_CATS_DIR):

if len(os.listdir(VALIDATION_DOGS_DIR)) > 0: for file in os.scandir(VALIDATION_DOGS_DIR):

os.remove(file.path)

os.remove(file.path)

os.remove(file.path)

os.remove(file.path)

TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/") VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")

TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/") VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/") # Empty directories in case you run this cell multiple times

# GRADED FUNCTION: split_data

--------------------------------------------------------------------------- FileNotFoundError Traceback (most recent call last)

print(f"\n\nOriginal cat's directory has {len(os.listdir(CAT_SOURCE_DIR))} images") print(f"Original dog's directory has {len(os.listdir(DOG_SOURCE_DIR))} images\n")

print(f"There are {len(os.listdir(TRAINING_CATS_DIR))} images of cats for training") print(f"There are {len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training") print(f"There are {len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation") print(f"There are {len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")

FileNotFoundError: [Errno 2] No such file or directory: '/tmp/cats-v-dogs/validation/cats/'

# Your function should perform copies rather than moving images so original directories should contain unchanged images

Now that you have successfully organized the data in a way that can be easily fed to Keras' ImageDataGenerator , it is time for you to code the generators that will yield batches of images, both for training and validation. For this, complete the train_val_generators function below. Something important to note is that the images in this dataset come in a variety of resolutions. Luckily, the flow_from_directory method allows you to standarize this by defining a tuple called target_size that will be used to convert each image to this target resolution. **For th**

Don't use data augmentation by setting extra parameters when you instantiate the ImageDataGenerator class. This will make the training of your model to take longer to reach the necessary accuracy threshold to pass this assignment and this topic will be covered in the next week.

<ipython-input-24-611c0d7ee290> in <cell line: 25>() 23 for file in os.scandir(TRAINING_DOGS_DIR):

# NOTE: Messages about zero length images should be printed out

# Check that the number of images matches the expected output

split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size) split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

> ---> 25 if len(os.listdir(VALIDATION_CATS_DIR)) > 0: 26 for file in os.scandir(VALIDATION_CATS_DIR):

24 os.remove(file.path)

# Define proportion of images used for training

split_size = .9 # Run the function

27 os.remove(file.path)

SEARCH STACK OVERFLOW

666.jpg is zero length, so ignoring. 11702.jpg is zero length, so ignoring.

Original cat's directory has 12500 images Original dog's directory has 12500 images

There are 11249 images of cats for training There are 11249 images of dogs for training There are 1250 images of cats for validation There are 1250 images of dogs for validation

**exercise, use a target_size of (150, 150)**.

# GRADED FUNCTION: train_val_generators

def train_val_generators(TRAINING_DIR, VALIDATION_DIR):

Creates the training and validation data generators

TRAINING_DIR (string): directory path containing the training images

train_generator, validation_generator - tuple containing the generators

VALIDATION_DIR (string): directory path containing the testing/validation images

**Hint:**

"""

Args:

Returns:

### START CODE HERE

"""

# grader-required-cell

**Expected Output:**

# Training and validation splits

# Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)

# Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)

train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

One last step before training is to define the architecture of the model that will be traine Complete the create_model function below which should return a Keras' Sequential model.

**Note that you should use at least 3 convolution layers to achieve the desired performance.**

batch_size=None, class_mode=None, target_size=(None, None))

Aside from defining the architecture of the model, you should also compile it so make sure to use a loss function that is compatible with t class_mode you defined in the previous exercise, which should also be compatible with the output of your network. You can tell if they aren

batch_size=None, class_mode=None, target_size=(None, None))

# Pass in the appropriate arguments to the flow_from_directory method train_generator = train_datagen.flow_from_directory(directory=None,

# Pass in the appropriate arguments to the flow_from_directory method validation_generator = validation_datagen.flow_from_directory(directory=None,

train_datagen = None

validation_datagen = None

return train_generator, validation_generator

Found 22498 images belonging to 2 classes. Found 2500 images belonging to 2 classes.

compatible if you get an error during training.

# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS

metrics=['accuracy'])

**Note:** You can ignore the UserWarning: Possibly corrupt EXIF data. warnings.

### END CODE HERE

# grader-required-cell # Test your generators

# grader-required-cell

### START CODE HERE

def create_model():

None,

### END CODE HERE

return model

])

# GRADED FUNCTION: create_model

# USE AT LEAST 3 CONVOLUTION LAYERS

model = tf.keras.models.Sequential([

model.compile(optimizer=None, loss=None,

Now it is time to train your model!

# Note that this may take some time. history = model.fit(train_generator,

epochs=15,

# Get the untrained model model = create_model()

# Train the model

**Expected Output:**

verbose=1,

# sets for each training epoch

acc=history.history['accuracy'] val_acc=history.history['val_accuracy']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------ # Plot training and validation accuracy per epoch #----------------------------------------------- plt.plot(epochs, acc, 'r', "Training Accuracy") plt.plot(epochs, val_acc, 'b', "Validation Accuracy") plt.title('Training and validation accuracy')

#------------------------------------------------ # Plot training and validation loss per epoch #----------------------------------------------- plt.plot(epochs, loss, 'r', "Training Loss") plt.plot(epochs, val_loss, 'b', "Validation Loss")

loss=history.history['loss'] val_loss=history.history['val_loss']

plt.show() print("")

plt.show()

def download_history(): import pickle

download_history()

from google.colab import files

files.download('history.pkl')

# Download metadata checker

import colab_metadata_checker

with open('history.pkl', 'wb') as f: pickle.dump(history.history, f)

Download your notebook for grading

#----------------------------------------------------------- # Retrieve a list of list results on training and test data

#-----------------------------------------------------------

validation_data=validation_generator)

Once training has finished, you can run the following cell to check the training and validation accuracy achieved at the end of each epoc **To pass this assignment, your model should achieve a training accuracy of at least 95% and a validation accuracy of at least 80%**. If your model didn't achieve these thresholds, try training again with a different model architecture and remember to use at least 3 convolutional layers.

You will probably encounter that the model is overfitting, which means that it is doing a great job at classifying the images in the training set b

Before downloading this notebook and closing the assignment, be sure to also download the history.pkl file which contains the informati

Along with the history.pkl file, you will also need to submit your solution notebook for grading. The following code cells will check if th notebook's grader metadata (i.e. hidden data in the notebook needed for grading) is not modified by your workspace. This will ensure that t

*if the metadata is missing*: A new notebook with your solutions will be created on this Colab workspace. It should be downloaded

-> Download .ipynb. You can name it anything you want as long as it is a valid .ipynb (jupyter notebook) fil

*if the metadata is intact*: Download the current notebook. Click on the File tab on the upper left corner of the screen then click on Download

struggles with new data. This is perfectly fine and you will learn how to mitigate this issue in the upcoming wee

of the training history of your model. You can download this file by running the cell belo

autograder can evaluate your code properly. Depending on its output, you will either:

!wget -nc https://storage.googleapis.com/tensorflow-1-public/colab_metadata_checker.py

# Please see the output of this cell to see which file you need to submit to the grader

automatically and you can submit that to the grader.

colab_metadata_checker.run('C2W1_Assignment_fixed.ipynb')

**Please disregard the following note if the notebook metadata is detected**

*Right-click on that file to save locally and submit it to the grade*

**Congratulations on finishing this week's assignmen**

needed to pre-process the images!

**Keep it up!**

*Note: Just in case the download fails for the second point above, you can also do these steps: Click the Folder icon on the left side of this screen to open the File Manager.*

*Click the Folder Refresh icon in the File Manager to see the latest files in the workspace. You should see a file ending with a _fixed.ipy*

You have successfully implemented a convolutional neural network that classifies images of cats and dogs, along with the helper functio

