import cv2
import numpy as np

def nlm_filtering(
    img: np.uint8,
    intensity_variance: float,
    patch_size: int,
    window_size: int,
) -> np.uint8:
    """
    Homework 2 Part 4
    Compute the filtered image given an input image, kernel size of image patch, spatial variance, and intensity range variance
    """
    img = img / 255
    img = img.astype("float32")
    H, W = img.shape
    img_filtered = np.zeros(img.shape)  # Placeholder for the filtered image

    n = (patch_size - 1) // 2  # Half size of the patch
    N = (window_size - 1) // 2  # Half size of the search window
    pad_size = n + N  # Total padding needed
    img_padded = np.pad(img, pad_size, mode='constant')  # Zero-padding

    # Loop over each pixel in the original image
    for i in range(H):
        for j in range(W):
            i_padded = i + pad_size
            j_padded = j + pad_size

            # Reference patch centered at [i_padded, j_padded]
            P_p = img_padded[i_padded - n: i_padded + n + 1, j_padded - n: j_padded + n + 1]

            w_total = 0  # Sum of weights
            I_total = 0  # Sum of weighted intensities

            # Loop over the search window
            for k in range(i_padded - N, i_padded + N + 1):
                for l in range(j_padded - N, j_padded + N + 1):
                    # Comparison patch centered at [k, l]
                    P_q = img_padded[k - n: k + n + 1, l - n: l + n + 1]

                    # Compute the squared difference between patches
                    D = np.sum((P_p - P_q) ** 2)

                    # Compute the weight
                    w = np.exp(-D / intensity_variance)

                    # Accumulate the weight and the weighted intensity
                    w_total += w
                    I_total += w * img_padded[k, l]

            # Compute the filtered pixel value
            img_filtered[i, j] = I_total / w_total

    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered


if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0)  # read gray image
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)  # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img)  # save image

    # Generate Gaussian noise
    noise = np.random.normal(0, 0.6, img.size)
    noise = noise.reshape(img.shape[0], img.shape[1]).astype('uint8')

    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)

    # NLM Filtering with fine-tuned parameters
    intensity_variance = 1  
    patch_size = 5  # Adjusted patch size
    window_size = 15  # Adjusted window size for better filtering
    img_nlm = nlm_filtering(img_noise, intensity_variance, patch_size, window_size)
    cv2.imwrite('results/im_nlm.png', img_nlm)