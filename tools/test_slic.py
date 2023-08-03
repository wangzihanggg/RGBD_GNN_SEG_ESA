import cv2
import numpy as np

# 读取图片
img_path = '/home/wangzihanggg/lena.jpg'
img = cv2.imread(img_path)

# 转换为Lab颜色空间
# lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
lab_img = img

# 运行SLIC超像素分割算法
compactness = 10      # 超像素紧密度，可以根据需要进行调整
slic = cv2.ximgproc.createSuperpixelSLIC(lab_img, region_size=10, ruler=compactness)
slic.iterate(num_iterations=10)
labels = slic.getLabels()

# 将超像素标签转换为3D图像形式
segmented_img = np.zeros_like(img)
for label in np.unique(labels):
    mask = labels == label
    segmented_img[mask] = np.mean(lab_img[mask], axis=0)

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('SLIC Segmentation', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
