import cv2
import numpy as np

# Load the image
img = cv2.imread('inp.png')




# Apply a Gaussian blur
blurred_img = cv2.GaussianBlur(np.clip(img + np.random.normal(0, 50, img.shape),
                                       0, 255).astype(np.uint8), (5, 5), 1)
# Show the images
cv2.imshow('Original Image', img)
cv2.imshow('Noisy Image', blurred_img)
# cv2.imshow('Blurred Image', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save the output image
blurred_img = cv2.resize(blurred_img, (256, 256))
cv2.imwrite('noise_img_50.png', blurred_img)
