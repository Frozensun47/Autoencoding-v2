from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
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


# Create a directory to save the output images of each layer
create_dir('outputimgs')


# Define the directories containing the input images
input_dir = 'Images'

# Create data generators to load and preprocess the images
datagen = ImageDataGenerator()
batch_size = 8
generator = datagen.flow_from_directory(
    input_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='input',
    shuffle=True
)
steps_per_epoch = generator.n // batch_size
print("length of input data" , len(generator))
num_samples = generator.samples
num_classes = generator.num_classes
input_shape = generator.image_shape
print(num_classes,num_samples,input_shape)

# Define the original model
input_layer = Input(shape=(256, 256, 3))
conv1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
               activation=activation_type, padding=padding_type)(input_layer)
conv2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
               activation=activation_type, padding=padding_type)(conv1)
conv3 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
               activation=activation_type, padding=padding_type)(conv2)
conv4 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
               activation=activation_type, padding=padding_type)(conv3)
upsamp1 = UpSampling2D(size=(2, 2))(conv4)
conv5 = Conv2D(64, kernel_size=(3, 3), activation=activation_type,
               padding=padding_type)(upsamp1)
upsamp2 = UpSampling2D(size=(2, 2))(conv5)
conv6 = Conv2D(64, kernel_size=(3, 3), activation=activation_type,
               padding=padding_type)(upsamp2)
upsamp3 = UpSampling2D(size=(2, 2))(conv6)
conv7 = Conv2D(64, kernel_size=(3, 3), activation=activation_type,
               padding=padding_type)(upsamp3)
upsamp4 = UpSampling2D(size=(2, 2))(conv7)
conv8 = Conv2D(3, kernel_size=(3, 3), activation=activation_type,
               padding=padding_type)(upsamp4)

model = Model(inputs=input_layer, outputs=conv8)

#resume trainng
model.load_weights('my_model.h5')

# Load an image
img = cv2.imread('inp.png')
img = cv2.resize(img, (256, 256))
x = np.expand_dims(img, axis=0)

# Set the loss function to be mean squared error (MSE)
loss_function = MeanSquaredError()

# Compile the model with the MSE loss
model.compile(optimizer='adam', loss=loss_function)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# Train the model using the training data
history = model.fit(generator, epochs=30, verbose=1)

# Plot the training loss over the epochs
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Save the model weights and architecture
model.save('my_model.h5')

# Create a new model that outputs the activations of each layer till layer 4
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Predict the output and get the activations of each layer till layer 4
activations = activation_model.predict(x)

# Save the output image and the activations of each layer
for i, activation in enumerate(activations):
    # Save the activation of each layer as a separate image till layer 4
    activation = np.squeeze(activation, axis=0)
    for j in range(activation.shape[-1]):
        cv2.imwrite(
            f'outputimgs/activation_layer_{i+1}_channel_{j+1}.jpg', activation[:, :, j])

# Predict the output of the entire model
output = model.predict(x)

# Save the output image
cv2.imwrite(f'output_image.jpg', output[0])
