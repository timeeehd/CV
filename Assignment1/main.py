import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# y = []
#
# for i in range(0,100):
#     x = random.random()
#     y.append(x ** 2)
#
# y = [random.random() for _ in range(100)]
# A = np.array(y*100).reshape(100,100)
#
# plt.imshow(A)
# plt.show()
# plt.imshow(A.astype(int))
# plt.show()

# img = cv2.imread('lab1a.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
#
# img = cv2.imread('lab1a.png', cv2.IMREAD_GRAYSCALE)
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img, cmap='gray')
# plt.show()
#
# img = cv2.imread('lab1a.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12, 8))
#
# ax1.imshow(img[..., 0])
# ax2.imshow(img[..., 1])
# ax3.imshow(img[..., 2])
# ax4.imshow(img)
#
# plt.show()
#
#
# img = cv2.imread('lab1a.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

# 0, 50, 100, 150, 200

img_original = cv2.imread('lab1a.png', cv2.IMREAD_COLOR)
grayImage = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
#
# fig, ((ax, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2,3, figsize=(12, 8))
# ax.imshow(img)
# ax.set_title('Original')
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY)
# ax1.imshow(blackAndWhiteImage, cmap='gray')
# ax1.set_title('BlackWhite threshold 0')
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 50, 255, cv2.THRESH_BINARY)
# ax2.imshow(blackAndWhiteImage, cmap='gray')
# ax2.set_title('BlackWhite threshold 50')
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 100, 255, cv2.THRESH_BINARY)
# ax3.imshow(blackAndWhiteImage, cmap='gray')
# ax3.set_title('BlackWhite threshold 100')
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 150, 255, cv2.THRESH_BINARY)
# ax4.imshow(blackAndWhiteImage, cmap='gray')
# ax4.set_title('BlackWhite threshold 150')
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
# ax4.imshow(blackAndWhiteImage, cmap='gray')
# ax4.set_title('BlackWhite threshold 200')
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 250, 255, cv2.THRESH_BINARY)
# ax5.imshow(blackAndWhiteImage, cmap='gray')
# ax5.set_title('BlackWhite threshold 250')
# plt.show()
img = img[380:430, 280:380]
fig, ((ax, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(12, 8))
ax.imshow(img)
ax.set_title('Original')
img1 = cv2.resize(img, (50, 50), interpolation=cv2.INTER_NEAREST)
ax1.imshow(img1)
ax1.set_title('Nearest')
img2 = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
ax2.imshow(img2)
ax2.set_title('Linear')
img3 = cv2.resize(img, (200, 100), interpolation=cv2.INTER_NEAREST)
ax3.imshow(img3)
ax3.set_title('Nearest')
img4 = cv2.resize(img, (200, 100), interpolation=cv2.INTER_LINEAR)
ax4.imshow(img4)
ax4.set_title('Linear')
img5 = cv2.resize(img, (200, 100), interpolation=cv2.INTER_CUBIC)
ax5.imshow(img5)
ax5.set_title('Cubic')
plt.show()
