import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.spatial.distance import cdist


def add_salt_pepper(image, amount):
    noisy_image = np.copy(image)
    mask = np.random.choice([0, 1, 2], size=image.shape, p=[1 - 2 * amount, amount, amount])
    noisy_image[mask == 1] = 255   # sól
    noisy_image[mask == 2] = 0     # pieprz
    return noisy_image

def mean_self(image, size):
    r = size // 2
    filtered_image = np.copy(image).astype(float)
    for i in range(r, image.shape[0] - r):
        for j in range(r, image.shape[1] - r):
            window = image[i - r:i + r + 1, j - r:j + r + 1]
            filtered_image[i, j] = np.mean(window)
    return filtered_image

def median_self(image, size):
    r = size // 2
    filtered_image = np.copy(image)
    for i in range(r, image.shape[0] - r):
        for j in range(r, image.shape[1] - r):
            window = image[i - r:i + r + 1, j - r:j + r + 1]
            filtered_image[i, j] = np.median(window, axis=(0, 1))
    return filtered_image

def vector_median_self(image, size):
    r = size // 2
    filtered_image = np.copy(image)
    for i in range(r, image.shape[0] - r):
        for j in range(r, image.shape[1] - r):
            window = image[i - r:i + r + 1, j - r:j + r + 1]
            colors = window.reshape(-1, 3)

            distances = cdist(colors, colors)
            total_distance = np.sum(distances, axis=1)

            best = np.argmin(total_distance)
            filtered_image[i, j] = colors[best]
    return filtered_image


#Zadanie_1:
chelsea = skimage.data.chelsea()

np.random.seed(42)

chelsea = skimage.data.chelsea()

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

chelsea_red = chelsea[::3, ::3, 0]
axs[0].imshow(chelsea_red, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('czerwony kanał')

chelsea_red_noisy = add_salt_pepper(chelsea_red, 0.1)
axs[1].imshow(chelsea_red_noisy, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('szum impulsowy')

chelsea_red_mean = mean_self(chelsea_red_noisy, 9)
axs[2].imshow(chelsea_red_mean, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('filtr uśredniający')

fig.tight_layout()
fig.savefig('Zad_1.png')

#Zadanie_2:

fig2, axs2 = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)

chelsea_small = chelsea[::3, ::3]

noise_levels = [0.1, 0.2, 0.3]
sizes = [3, 5, 9]

noisy_images = []
median_9_images = []

for row, level in enumerate(noise_levels):
    noisy = add_salt_pepper(chelsea_small, level)
    noisy_images.append(noisy)

    axs2[row, 0].imshow(noisy)
    axs2[row, 0].set_ylabel(f'noise level = {level}')

    for col, size in enumerate(sizes):
        filtered = median_self(noisy, size)
        axs2[row, col + 1].imshow(filtered)
        if size == 9:
            median_9_images.append(filtered)

axs2[0, 0].set_title('zaszumiony')
for col, size in enumerate(sizes):
    axs2[0, col + 1].set_title(f'filtr medianowy | k = {size}')

fig2.tight_layout()
fig2.savefig('Zad_2.png')

#Zadanie_3:

fig3, axs3 = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)

noise_levels = [0.1, 0.2, 0.3]
noisy_imgs = []


for row, level in enumerate(noise_levels):
    noisy = add_salt_pepper(chelsea_small, level)
    noisy_imgs.append(noisy)
    axs3[row, 0].imshow(noisy)
    axs3[row, 0].set_ylabel(f'noise level = {level}')

    filtered = median_self(noisy, 9)
    axs3[row, 1].imshow(filtered)

    filtered_vector = vector_median_self(noisy, 7)
    axs3[row, 2].imshow(filtered_vector)

    axs3[0, 0].set_title('zaszumiony')
    axs3[0, 1].set_title('filtr medianowy')
    axs3[0, 2].set_title('wektorowy filtr medianowy')

fig3.tight_layout()
fig3.savefig('Zad_3.png')

plt.show()