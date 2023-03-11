import cv2
import numpy as np

# Load the image
img = cv2.imread('inp.png')



# Add some random noise
noisy_img = np.clip(img + np.random.normal(0, 50, img.shape),
                    0, 255).astype(np.uint8)

# Apply a Gaussian blur
# blurred_img = cv2.GaussianBlur(noisy_img, (5, 5), 2)
# Show the images
cv2.imshow('Original Image', img)
cv2.imshow('Noisy Image', noisy_img)
# cv2.imshow('Blurred Image', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save the output image
cv2.imwrite('noise_img.png', noisy_img)
