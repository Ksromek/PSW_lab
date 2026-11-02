import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage


#Zadanie_1:

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

chelsea = skimage.data.chelsea()

chelsea_mono = np.mean(chelsea, axis=2)

sobel_s1 = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])

correlate_chelsea = ndimage.correlate(chelsea_mono, sobel_s1)

convolve_chelsea = ndimage.convolve(chelsea_mono, sobel_s1)

axs[0].imshow(correlate_chelsea, cmap='gray')
axs[0].set_title('Korelacja scipy')
axs[1].imshow(convolve_chelsea, cmap='gray')
axs[1].set_title('Konwolucja scipy')

fig.tight_layout()
fig.savefig("Zad_1.png")

#Zadanie_2:

def correlate_self(image, kernel):
    correlated_image = np.zeros_like(image, dtype=float)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            pixel = 0
            for u in range(-1, 2):
                for v in range(-1, 2):
                    pixel = pixel + kernel[u + 1, v + 1] * image[i + u, j + v]
            correlated_image[i, j] = pixel
    return correlated_image


def convolve_self(image, kernel):
    return correlate_self(image, np.flip(kernel))

fig2, axs2 = plt.subplots(3, 2, figsize=(12, 12), sharex=True, sharey=True)

axs2[0, 0].imshow(correlate_chelsea, cmap='gray')
axs2[0, 0].set_title('Korelacja scipy')
axs2[0, 1].imshow(convolve_chelsea, cmap='gray')
axs2[0, 1].set_title('Konwolucja scipy')

correlate_own = correlate_self(chelsea_mono, sobel_s1)
convolve_own = convolve_self(chelsea_mono, sobel_s1)

axs2[1, 0].imshow(correlate_own, cmap='gray')
axs2[1, 0].set_title('Korelacja własna')
axs2[1, 1].imshow(convolve_own, cmap='gray')
axs2[1, 1].set_title('Konwolucja własna')

axs2[2, 0].imshow(np.abs(correlate_chelsea - correlate_own), cmap='gray')
axs2[2, 0].set_title('Różnica korelacja własna vs scipy')
axs2[2, 1].imshow(np.abs(convolve_chelsea - convolve_own), cmap='gray')
axs2[2, 1].set_title('Różnica konwolucja własna vs scipy')

fig2.tight_layout()
fig2.savefig("Zad_2.png")

#Zadanie_3:
fig3, axs3 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

axs3[0, 0].imshow(chelsea_mono, cmap='gray', vmin=0, vmax=255)
axs3[0, 0].set_title('Oryginał')

def blur_self(image, kernel):
    r = kernel.shape[0] // 2
    blur_image = np.zeros_like(image, dtype=float)
    for i in range(r, image.shape[0] - r):
        for j in range(r, image.shape[1] - r):
            pixel = 0
            for u in range(-r, r + 1):
                for v in range(-r, r + 1):
                    pixel = pixel + kernel[u + r, v + r] * image[i + u, j + v]
            blur_image[i, j] = pixel
    return blur_image


blur_kernel = np.ones((7, 7)) / 49
chelsea_blurred = blur_self(chelsea_mono, blur_kernel)

axs3[0, 1].imshow(chelsea_blurred, cmap='gray', vmin=0, vmax=255)
axs3[0, 1].set_title('Rozmycie')

mask = chelsea_mono - chelsea_blurred

axs3[1, 0].imshow(mask, cmap='gray')
axs3[1, 0].set_title('Maska')

unsharp_chelsea = chelsea_mono + mask
axs3[1, 1].imshow(unsharp_chelsea, cmap='gray', vmin=0, vmax=255)
axs3[1, 1].set_title('Unsharp masking')

fig3.tight_layout()
fig3.savefig("Zad_3.png")
plt.show()