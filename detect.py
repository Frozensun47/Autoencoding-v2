from tensorflow.keras.models import load_model
import cv2
import numpy as np
# Load the saved model
model = load_model('weights/my_model.h5')

# Load an image
img = cv2.imread('noise_img_50.png')
img = cv2.resize(img, (256, 256))
x = np.expand_dims(img, axis=0)/255

# Predict the output of the entire model
output = model.predict(x)*255
# Save the output image
cv2.imwrite(f'output_image_detect.jpg', output[0])
