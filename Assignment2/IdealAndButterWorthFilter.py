import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_ideal_filter(img, D_0=50, filter_type='low_pass'):
    M, N = img.shape
    mid_U = M / 2
    mid_V = N / 2
    U = np.arange(0, M, 1)
    V = np.arange(0, N, 1)
    U, V = np.meshgrid(U, V)
    H = np.sqrt((U - mid_U) ** 2 + (V - mid_V) ** 2)
    if filter_type == 'low_pass':
        H = H <= D_0
    else:
        H = H >= D_0
    return H


def get_butterworth_filter(img, D_0=50, filter_type='high_pass', n=1):
    M, N = img.shape
    mid_U = M / 2
    mid_V = N / 2
    U = np.arange(0, M, 1)
    V = np.arange(0, N, 1)
    U, V = np.meshgrid(U, V)
    tmp_d = np.sqrt((U - mid_U) ** 2 + (V - mid_V) ** 2)
    if filter_type == 'high_pass':
        H = 1 / (1 + np.power((D_0 / tmp_d), 2 * n))
    else:
        H = 1 / (1 + np.power((tmp_d / D_0), 2 * n))
    return H


if __name__ == "__main__":
    img = cv2.imread("lab2b.png", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 8))

    H = get_butterworth_filter(img, filter_type='low_pass')
    ax.imshow(H, cmap='gray')
    H = get_butterworth_filter(img)
    ax1.imshow(H, cmap='gray')
    H = get_ideal_filter(img, filter_type='low_pass')
    ax2.imshow(H, cmap='gray')
    H = get_ideal_filter(img, filter_type='high_pass')
    ax3.imshow(H, cmap='gray')
    plt.show()

