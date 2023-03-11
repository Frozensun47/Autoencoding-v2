from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D,Add
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


padding_type = 'same'
activation_type = 'relu'

# Define a function to create a directory


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Create a directory to save the output images
output_dir = 'outputimgs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Define the directories containing the input images
input_dir = 'Images'

# Create data generators to load and preprocess the images
datagen = ImageDataGenerator(
    # preprocessing_function=lambda x: np.clip(x + np.random.normal(0, 10, x.shape),0, 255).astype(np.uint8)
)
batch_size = 4
generator = datagen.flow_from_directory(
    input_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='input',
    shuffle=True
)
steps_per_epoch = generator.n // batch_size
# print("length of input data" , len(generator))
# num_samples = generator.samples
# num_classes = generator.num_classes
# input_shape = generator.image_shape
# print(num_classes,num_samples,input_shape)


# Define the original model with skip connections
input_img = Input(shape=(256, 256, 3))

# # Encoder
# conv = Conv2D(64, kernel_size=(3, 3), activation='relu',
#                padding='same')(input_img)
# pool = MaxPooling2D(pool_size=(2, 2))(conv)
# conv1 = Conv2D(64, kernel_size=(3, 3),
#                activation='relu', padding='same')(pool) #128
# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
# conv2 = Conv2D(64, kernel_size=(3, 3),
#                activation='relu', padding='same')(pool1) #256
# pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
# conv3 = Conv2D(64, kernel_size=(3, 3),
#                activation='relu', padding='same')(pool2)#512
# pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
# conv4 = Conv2D(64, kernel_size=(3, 3),
#                activation='relu', padding='same')(pool3)#1024

# Encoder
conv = Conv2D(8, kernel_size=(3, 3), activation='relu',
              padding='same')(input_img)
conv1 = Conv2D(16, kernel_size=(3,3),
               strides=(2, 2), activation='relu', padding='same')(conv)  # 128
conv2 = Conv2D(32, kernel_size=(3, 3),
               strides=(2, 2), activation='relu', padding='same')(conv1)  # 256
conv3 = Conv2D(64, kernel_size=(3, 3),
               strides=(2, 2), activation='relu', padding='same')(conv2)  # 512
conv4 = Conv2D(64, kernel_size=(3, 3),
               strides=(2, 2), activation='relu', padding='same')(conv3)  # 1024


# Decoder
up = UpSampling2D(size=(2, 2))(conv4)
conv5 = Conv2D(64, kernel_size=(3, 3), activation='relu',
               padding='same')(up)  # 512
add1 = Add()([conv3, conv5])
conv6 = Conv2D(64, kernel_size=(3, 3),
               activation='relu', padding='same')(add1)  # 512

up1 = UpSampling2D(size=(2, 2))(conv6)
conv7 = Conv2D(32, kernel_size=(3, 3), activation='relu',
               padding='same')(up1)  # 256
add2 = Add()([conv2, conv7])
conv8 = Conv2D(32, kernel_size=(3, 3),
               activation='relu', padding='same')(add2)  # 256

up2 = UpSampling2D(size=(2, 2))(conv8)
conv9 = Conv2D(16, kernel_size=(3,3), activation='relu',
               padding='same')(up2)  # 128
add3 = Add()([conv1, conv9])
conv10 = Conv2D(16, kernel_size=(3, 3),
                activation='relu', padding='same')(add3)  # 128

up3 = UpSampling2D(size=(2, 2))(conv10)
conv11 = Conv2D(8, kernel_size=(3, 3), activation='relu',
                padding='same')(up3)  # 64
add4 = Add()([conv, conv11])
conv12 = Conv2D(8, kernel_size=(3, 3),
                activation='relu', padding='same')(add4)  # 64

output_img = Conv2D(3, kernel_size=(
    1, 1), activation='relu', padding='same')(conv12)

model = Model(inputs=input_img, outputs=output_img)



plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#resume trainng
model.load_weights('my_model.h5')

# Load an image
img = cv2.imread('inp.png')
img = cv2.resize(img, (256, 256))
x = np.expand_dims(img, axis=0)

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
history = model.fit(generator, epochs=10, steps_per_epoch=steps_per_epoch,
                    verbose=1, callbacks=[model_checkpoint_callback])

# Plot the training loss over the epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Save the model weights and architecture
model.save('my_model.h5')

# Create a new model that outputs the activations of each layer
layer_outputs = [
    layer.output for layer in model.layers if isinstance(layer, Conv2D)]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Predict the output and get the activations
activations = activation_model.predict(x)

# Save the output image and the activations of each layer
for i, activation in enumerate(activations):
    # Create a directory to save the activations of each layer
    layer_dir = os.path.join(output_dir, f'layer_{i+1}')
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)

    # Save the activation of each layer as a separate image
    activation = np.squeeze(activation, axis=0)
    for j in range(activation.shape[-1]):
        cv2.imwrite(
            os.path.join(layer_dir, f'channel_{j+1}.jpg'), activation[:, :, j])

# Predict the output of the entire model
output = model.predict(x)

# Save the output image
cv2.imwrite(f'output_image.jpg', output[0])
