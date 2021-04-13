import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from Assignment2.convolution import *


def contrast_stretching(img, a=50, b=150, ya=30, yb=200, alpha=0.3, beta=2, gamma=1, L=256):
    width, height = img.shape
    for x in range(width):
        for y in range(height):
            if 0 <= img[x, y] < a:
                img[x, y] = alpha * img[x, y]
            if a <= img[x, y] < b:
                img[x, y] = beta * (img[x, y] - a) + ya
            if b <= img[x, y] < L:
                img[x, y] = gamma * (img[x, y] - b) + yb
    return img


def clipping(img, a=50, b=150, L=256, beta=2):
    width, height = img.shape
    for x in range(width):
        for y in range(height):
            if 0 <= img[x, y] < a:
                img[x, y] = 0
            if a <= img[x, y] < b:
                img[x, y] = beta * (img[x, y] - a)
            if b <= img[x, y] < L:
                img[x, y] = beta * (b - a)
    return img


def equalization(img):
    width, height = img.shape
    color = np.zeros(255)
    for x in range(width):
        for y in range(height):
            color[img[x, y]] += 1

    total_pixels = width * height
    sum_color = np.zeros(255)
    for i in range(255):
        if i == 0:
            sum_color[i] = color[i]
        else:
            sum_color[i] = sum_color[i - 1] + color[i]
    for x in range(width):
        for y in range(height):
            pix = img[x, y]
            img[x, y] = round((sum_color[pix] / total_pixels) * 254)
    return img


def zero_padding(img, padded):
    width, height = img.shape
    zero_padded_image = np.zeros((width + 2 * padded, height + 2 * padded))
    zero_padded_image[padded:width + padded, padded:height + padded] = img
    return zero_padded_image


def convulational_2d(img, filter, padding):
    width, height = img.shape
    for x in range(padding, width - padding + 1):
        for y in range(padding, height - padding + 1):
            sum = 0
            for i in range(-4, 5):
                for j in range(-4, 5):
                    sum += (img[x + i, y + j] * filter[i + 4, j + 4])
            img[x, y] = sum
    return img


def convulational_horizontal(img, filter, padding):
    width, height = img.shape
    for x in range(padding, width - padding + 1):
        for y in range(padding, height - padding + 1):
            sum = 0
            for i in range(-4, 5):
                sum += (img[x + i, y] * filter[i + 4])
            img[x, y] = sum
    return img


def convulational_vertical(img, filter, padding):
    width, height = img.shape
    for x in range(padding, width - padding + 1):
        for y in range(padding, height - padding + 1):
            sum = 0
            for i in range(-4, 5):
                sum += (img[x, y + i] * filter[i + 4])
            img[x, y] = sum
    return img


def remove_padding(img, padded):
    width, height = img.shape
    return img[padded:width - padded, padded:height - padded]


# TODO:part a
# img = cv2.imread('lab2a.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# plt.imshow(img, cmap='gray')
#
# plt.show()
#
# im2 = contrast_stretching(img)
# plt.imshow(im2, cmap='gray')
# plt.show()
#
# im3 = clipping(img)
# plt.imshow(im2, cmap='gray')
# plt.show()

# TODO:part b
# img = cv2.imread('Unequalized_H.jpg', cv2.IMREAD_GRAYSCALE)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
# ax1.hist(img.flatten(),256,[0,256])
# ax2.imshow(img, cmap='gray')
# img2 = equalization(img)
# ax3.hist(img.flatten(),256,[0,256])
# ax4.imshow(img2, cmap='gray')
# plt.show()

# TODO: part c
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
# img = cv2.imread('lab2a.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ax1.imshow(img, cmap='gray')

# filter = np.ones((9, 9)) * 1 / 81
# zero_pad_img = zero_padding(img, 8)
# conv_img = convulational_2d(zero_pad_img, filter, 8)
# remove_padd_img = remove_padding(conv_img, 8)
# ax2.imshow(remove_padd_img, cmap='gray')
#
#
# filter_1 = np.ones(9) * 1/9
# zero_pad_img = zero_padding(img, 8)
# conv_img = convulational_horizontal(zero_pad_img, filter_1, 8)
# remove_padd_img = remove_padding(conv_img, 8)
# ax3.imshow(remove_padd_img, cmap='gray')
# filter_2 = np.transpose(filter_1)
# conv_img = convulational_vertical(zero_pad_img, filter_2, 8)
# remove_padd_img = remove_padding(conv_img, 8)
# ax4.imshow(remove_padd_img, cmap='gray')
# plt.show()

# TODO: part c.2
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
# img = cv2.imread('lab2a.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ax1.imshow(img, cmap='gray')
#
# filter = np.ones((9, 9)) * 1 / 81
# new_img = convolution(img, filter)
# ax2.imshow(new_img, cmap='gray')
#
# filter = np.ones(9)[np.newaxis]
# filter = filter * 1/9
# new_img = convolution(img, filter)
# ax3.imshow(new_img, cmap='gray')
#
# new_img2 = convolution(new_img, filter.T)
# ax4.imshow(new_img2, cmap='gray')
#
# plt.show()

# TODO: part d
# fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(12, 8))
# img = cv2.imread('lab2a.png', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ax1.imshow(img, cmap='gray')
#
# filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
# new_img = convolution(img, filter)
# ax2.imshow(new_img, cmap='gray')
# plt.show()

# TODO: part 2a
img = cv2.imread('lab2b.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fft_res = np.fft.fft2(img)
plt.imshow(np.log10(np.abs(fft_res)), cmap='gray')
plt.show()
fft_shift_res = np.fft.fftshift(fft_res)
plt.imshow(np.log10(np.abs(fft_shift_res)), cmap='gray')
plt.show()
fft_shift_ifft2 = np.fft.ifft2(fft_shift_res)
plt.imshow(np.log10(np.abs(fft_shift_ifft2)), cmap='gray')
plt.show()


