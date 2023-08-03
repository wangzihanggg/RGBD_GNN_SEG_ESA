import numpy as np

from skimage.io import imread
def reduce_to_single_channel(image):
    # 获取图像的高度、宽度和通道数
    h, w, c = image.shape

    # 创建一个空白图像，用于存储缩减后的单通道图像
    single_channel_image = np.zeros((h, w), dtype=np.uint8)

    # 对每个像素进行通道合并
    for i in range(h):
        for j in range(w):
            # 获取当前像素的RGB值
            r, g, b = image[i, j]

            # 取RGB通道的平均值，并将结果赋值给单通道图像
            single_channel_image[i, j] = np.mean([r, g, b])

    return single_channel_image


# 三通道的图片矩阵
# 假设image为一个[h, w, 3]的三通道图像矩阵
# 例如：image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
# 注意：这里示例中的image是随机生成的，实际应用中需要根据实际情况加载你的图像
image = imread('lena.jpg')

# 缩减为单通道图像
single_channel_image = reduce_to_single_channel(image)

# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(single_channel_image, cmap='gray')
plt.title('Reduced to Single Channel')
plt.axis('off')

plt.show()
