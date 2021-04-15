import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat

from convolution import convolution

img = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
# plt.imshow(img, cmap='gray')
# plt.show()
#
# Mx = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
# My = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
#
# gradientx = convolution(img, Mx)
# gradienty = convolution(img, My)
#
# magnitude = np.sqrt(gradientx * gradientx + gradienty * gradienty)
#
# # width, height = magnitude.shape
# # for i in range(width):
# #     for j in range(height):
# #         if magnitude[i][j] > 1:
# #             magnitude[i][j] = 1
# #         else:
# #             magnitude[i][j] = 0
# #
# t = .25
# lena_t = magnitude > t
#
# plt.imshow(lena_t, cmap='gray')
# plt.show()

log5 = loadmat('./Log5.mat')['Log5']
log17 = loadmat('./Log17.mat')['Log17']
laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]])

lena_lap3 = convolution(img,laplacian)
plt.imshow(lena_lap3, cmap='gray')
plt.show()
lena_lap5 = convolution(img,log5)
plt.imshow(lena_lap5, cmap='gray')
plt.show()
lena_lap17 = convolution(img,log17)
plt.imshow(lena_lap17, cmap='gray')
plt.show()

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(12, 8))

bin_img = (lena_lap17 >= 0).astype(int)
filter = np.array([[1,1,1], [1,1,1], [1,1,1]])
zero_cross = convolution(bin_img,filter)
# without removing padding
edge_img = np.zeros((zero_cross.shape[0], zero_cross.shape[1]))
width, height = zero_cross.shape
for i in range(width):
    for j in range(height):
        if bin_img[i,j] == 1:
            if zero_cross[i,j] < 9:
               edge_img[i,j] = 1

ax1.imshow(edge_img, cmap='gray')

# with removing padding, since you do padding 1 at all sides, 0,0 in the result img, is 1,1 in the original image
bin_img = (lena_lap17 >= 0).astype(int)
result = np.zeros((zero_cross.shape[0] - 2, zero_cross.shape[1] - 2))
width, height = zero_cross.shape
for i in range(1,width -1):
    for j in range(1,height-1):
        if bin_img[i,j] == 1:
            if zero_cross[i,j] < 9:
               result[i-1,j-1] = 1

ax2.imshow(result, cmap='gray')

# Solution code
zero_cross = np.zeros((lena_lap3.shape[0] - 2, lena_lap3.shape[1] - 2))
for i in range(1, bin_img.shape[0]-1):
    for j in range(1, bin_img.shape[1] - 1):
        if bin_img[i,j] == 1:
            tmp_window = bin_img[i-1:i+2, j-1:j+2]
            value = np.sum(tmp_window * filter)
            if value < 9:
                zero_cross[i-1,j-1] = 1
ax3.imshow(zero_cross, cmap='gray')
plt.show()
# zero_cross_edge = zero_cross < 0
