from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, Add, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


padding_type = 'same'
activation_type = 'LeakyReLU'

# Define a function to create a directory


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Create a directory to save the output images
output_dir_layer_imgs = 'output_layer_imgs'
if not os.path.exists(output_dir_layer_imgs):
    os.makedirs(output_dir_layer_imgs)


# Define the directories containing the input images
input_dir = 'Images\Input_parent'
output_dir = 'Images\Label_parent'

datagen = ImageDataGenerator(
    rescale=1./255,
)
label_datagen = ImageDataGenerator(
    rescale=1./255,
)

batch_size = 8

input_generator = datagen.flow_from_directory(
    input_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=10
)

label_generator = label_datagen.flow_from_directory(
    output_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=10
)

train_generator = zip(input_generator, label_generator)

steps_per_epoch = input_generator.n // batch_size
# print("length of input data" , len(generator))
# num_samples = generator.samples
# num_classes = generator.num_classes
# input_shape = generator.image_shape
# print(num_classes,num_samples,input_shape)


# Define the original model with skip connections
input_img = Input(shape=(256, 256, 3))


# Encoder
conv = Conv2D(8, kernel_size=(3, 3), activation='relu',
              padding='same')(input_img)
bn1 = BatchNormalization()(conv)
conv1 = Conv2D(16, kernel_size=(3, 3),
               strides=(2, 2), activation='relu', padding='same')(bn1)  # 128
bn2 = BatchNormalization()(conv1)
conv2 = Conv2D(32, kernel_size=(3, 3),
               strides=(2, 2), activation='relu', padding='same')(bn2)  # 256
bn3 = BatchNormalization()(conv2)
conv3 = Conv2D(64, kernel_size=(3, 3),
               strides=(2, 2), activation='relu', padding='same')(bn3)  # 512
bn4 = BatchNormalization()(conv3)
conv4 = Conv2D(64, kernel_size=(3, 3),
               strides=(2, 2), activation='relu', padding='same')(bn4)  # 1024


# Decoder
up = UpSampling2D(size=(2, 2))(conv4)
conv5 = Conv2D(64, kernel_size=(3, 3), activation='relu',
               padding='same')(up)  # 512
bn5 = BatchNormalization()(conv5)
add1 = Add()([conv3, bn5])
conv6 = Conv2D(64, kernel_size=(3, 3),
               activation='relu', padding='same')(add1)  # 512
bn6 = BatchNormalization()(conv6)

up1 = UpSampling2D(size=(2, 2))(bn6)
conv7 = Conv2D(32, kernel_size=(3, 3), activation='relu',
               padding='same')(up1)  # 256
bn7 = BatchNormalization()(conv7)
add2 = Add()([conv2, bn7])
conv8 = Conv2D(32, kernel_size=(3, 3),
               activation='relu', padding='same')(add2)  # 256
bn8 = BatchNormalization()(conv8)

up2 = UpSampling2D(size=(2, 2))(bn8)
conv9 = Conv2D(16, kernel_size=(3, 3), activation='relu',
               padding='same')(up2)  # 128
bn9 = BatchNormalization()(conv9)
add3 = Add()([conv1, bn9])
conv10 = Conv2D(16, kernel_size=(3, 3),
                activation='relu', padding='same')(add3)  # 128
bn10 = BatchNormalization()(conv10)

up3 = UpSampling2D(size=(2, 2))(bn10)
conv11 = Conv2D(8, kernel_size=(3, 3), activation='relu',
                padding='same')(up3)  # 64
bn11 = BatchNormalization()(conv11)
add4 = Add()([conv, bn11])
conv12 = Conv2D(8, kernel_size=(3, 3),
                activation='relu', padding='same')(add4)  # 64
bn12 = BatchNormalization()(conv12)

output_img = Conv2D(3, kernel_size=(
    1, 1), activation='relu', padding='same')(bn12)

model = Model(inputs=input_img, outputs=output_img)



plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#resume trainng
model.load_weights('weights/my_model.h5')

# Set the loss function to be mean squared error (MSE)
loss_function = MeanSquaredError()

checkpoint_filepath = 'weights/weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

# Compile the model with the MSE loss
model.compile(optimizer='adam', loss=loss_function)

# Train the model using the training data
history = model.fit(train_generator, epochs=20, steps_per_epoch=steps_per_epoch,
                    verbose=1, callbacks=[model_checkpoint_callback])

# Plot the training loss over the epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Save the model weights and architecture
model.save('weights/my_model.h5')

# Create a new model that outputs the activations of each layer
layer_outputs = [
    layer.output for layer in model.layers if isinstance(layer, Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Load an image
img = cv2.imread('noise_img.png')
img = cv2.resize(img, (256, 256))
x = np.expand_dims(img, axis=0)/255
cv2.imwrite(f'input_image.jpg', img)

# Predict the output and get the activations
activations = activation_model.predict(x)

# Save the output image and the activations of each layer
for i, activation in enumerate(activations):
    # Create a directory to save the activations of each layer
    layer_dir = os.path.join(output_dir_layer_imgs, f'layer_{i+1}')
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)

    # Save the activation of each layer as a separate image
    activation = np.squeeze(activation, axis=0)
    for j in range(activation.shape[-1]):
        cv2.imwrite(
            os.path.join(layer_dir, f'channel_{j+1}.jpg'), activation[:, :, j]*255)

# Predict the output of the entire model
output = model.predict(x)*255
# Save the output image
cv2.imwrite(f'output_image.jpg', output[0])
