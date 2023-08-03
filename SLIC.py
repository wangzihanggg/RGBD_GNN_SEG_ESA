import numpy as np
from skimage.segmentation import slic
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2

def getSuperpixelImage(input_image):
    gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    h, w, _ = input_image.shape
    n_superpixels_list = [2048, 1024, 512, 256]
    superpixel_images = np.zeros((len(n_superpixels_list), h, w), dtype=int)

    for idx, n_superpixels in enumerate(n_superpixels_list):
        segments = slic(input_image, n_segments=n_superpixels, compactness=10)
        # superpixel_image = np.zeros((h, w), dtype=np.float)
        # for label in np.unique(segments):
        #     mask = segments == label
        #     average_color = np.mean(gray[mask], axis=0)
        #     superpixel_image[mask] = average_color
        # superpixel_image = superpixel_image.astype(np.uint8)
        superpixel_images[idx] = segments
        # print(len(np.unique(segments)))
    return superpixel_images

if __name__ == '__main__':
    image_path = 'lena.jpg'
    input_image = imread(image_path)
    segmented_map = getSuperpixelImage(input_image)
    n_superpixels_list = [2048, 1024, 512, 256]
    # 可视化结果
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for idx, superpixel_image in enumerate(segmented_map):
        axes[idx].imshow(superpixel_image, cmap='nipy_spectral')
        axes[idx].set_title(f'{n_superpixels_list[idx]} Superpixels')
        axes[idx].axis('off')

    plt.show()
