import cv2
import numpy as np
import math

def bilateral_filtering(
    img: np.uint8,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.uint8:
    """
    Homework 2 Part 3
    Compute the bilaterally filtered image given an input image, kernel size, spatial variance, and intensity range variance.
    """
    # Normalize input image
    img = img / 255.0
    img = img.astype("float32")
    img_filtered = np.zeros_like(img)

    # Create zero padding around the image
    padding = kernel_size // 2
    img_padded = np.pad(img, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    
    # Precompute the spatial Gaussian kernel
    spatial_gaussian = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - kernel_size // 2
            y = j - kernel_size // 2
            spatial_gaussian[i, j] = np.exp(-(x**2 + y**2) / (2 * spatial_variance))
    
    sizeX, sizeY = img.shape

    # Apply bilateral filter to each pixel
    for i in range(sizeX):
        for j in range(sizeY):
            # Extract local window
            local_window = img_padded[i:i+kernel_size, j:j+kernel_size]
            
            # Compute intensity Gaussian weights
            intensity_gaussian = np.exp(-(local_window - img[i, j])**2 / (2 * intensity_variance))
            
            # Combine spatial and intensity weights
            bilateral_weights = spatial_gaussian * intensity_gaussian
            
            # Normalize the weights
            bilateral_weights /= np.sum(bilateral_weights)
            
            # Apply the filter
            img_filtered[i, j] = np.sum(local_window * bilateral_weights)

    # Convert the result back to uint8 format
    img_filtered = np.uint8(img_filtered * 255)
    return img_filtered


if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0, 0.6, img.size)
    noise = noise.reshape(img.shape[0], img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    spatial_variance = 30  # sigma_s^2
    intensity_variance = 0.5  # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)
