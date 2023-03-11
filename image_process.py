import os
import numpy as np
from PIL import Image

# Define the directory paths
input_dir = 'Images/Train'
output_dir = 'Images/Train_processed'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the lambda function


def func(x): return np.clip(
    x + np.random.normal(0, 20, x.shape), 0, 255)


# Loop through the images in the input directory
for filename in os.listdir(input_dir):
    # Load the image
    img = Image.open(os.path.join(input_dir, filename))

    # Apply the lambda function to the image
    img_processed = Image.fromarray(np.uint8(func(np.array(img))))

    # Save the processed image to the output directory
    img_processed.save(os.path.join(output_dir, filename))
