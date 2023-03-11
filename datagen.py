import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
import cv2

# Define the directories containing the input images
input_dir = 'Images\Input_parent'
output_dir = 'Images\Label_parent'

datagen = ImageDataGenerator(
    rescale=1./255,
)
label_datagen = ImageDataGenerator(
    rescale=1./255,
)

batch_size = 4

input_generator = datagen.flow_from_directory(
    input_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=7
)

label_generator = label_datagen.flow_from_directory(
    output_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=7
)

train_generator = zip(input_generator, label_generator)

# Get the first batch of data from the generator
batch = next(train_generator)

# Get the input and label images from the batch
input_images = batch[0][0]
label_images = batch[1][0]

# Rescale the pixel values of the input and label images from [0, 1] to [0, 255]
input_images = np.uint8(input_images * 255)
label_images = np.uint8(label_images * 255)

# Display the first input and label images
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].imshow(input_images)
axes[0].set_title('Input Image')
axes[0].axis('off')
axes[1].imshow(label_images)
axes[1].set_title('Label Image')
axes[1].axis('off')
plt.show()
