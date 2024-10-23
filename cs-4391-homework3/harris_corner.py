"""
CS 4391 Homework 3 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


#TODO: implement this function
# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with values 0 and 1, where 1s indicate corners of the input image 
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):
    height, width = R.shape # Get the height and width of the input matrix
    mask = np.zeros((height, width), dtype=np.uint8) 

    for i in range(1, height - 1): # Iterate through the image
        for j in range(1, width - 1): 
            local_patch = R[i-1:i+2, j-1:j+2] # Get the 3x3 local patch around the current pixel
            max_value = np.max(local_patch) # Get the maximum value in the local patch

            # If the current value is the maximum, mark it as a corner
            if R[i, j] == max_value and R[i, j] > 0:
                mask[i, j] = 1

    return mask 

#TODO: implement this function
# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with values 0 and 1, where 1s indicate corners of the input image
# Follow the steps in Lecture 7 slides 26-27
# You can use openCV functions and numpy functions
def harris_corner(im):
    # Step 0: Convert RGB to Grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    gray = np.float32(gray) 

    # Step 1: Compute image gradient using Sobel filters
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Step 2: Compute products of derivatives
    Ixx = Ix * Ix 
    Iyy = Iy * Iy 
    Ixy = Ix * Iy 

    # Step 3: Compute the sums of products of derivatives using Gaussian filter
    Ixx = cv2.GaussianBlur(Ixx, (5, 5), sigmaX=1.5) 
    Iyy = cv2.GaussianBlur(Iyy, (5, 5), sigmaX=1.5) 
    Ixy = cv2.GaussianBlur(Ixy, (5, 5), sigmaX=1.5) 

    # Step 4: Compute determinant and trace of the M matrix
    det_M = (Ixx * Iyy) - (Ixy ** 2) 
    trace_M = Ixx + Iyy 

    # Step 5: Compute R scores
    k = 0.05 
    R = det_M - k * (trace_M ** 2) 

    # Step 6: Thresholding
    threshold = 0.01 * R.max() 
    R[R < threshold] = 0

    # Step 7: Non-maximum suppression
    corner_mask = non_maximum_suppression(R) 

    return corner_mask 


# main function
if __name__ == '__main__':

    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)
    
    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)
    
    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()
        
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('our corner image')
    
    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('opencv corner image')

    plt.show()
