import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage

# Zadanie 1: Szum sól-pieprz
# Wczytaj barwny obraz chelsea i dodaj szum sól-pieprz za pomocą np.random.choice
# (90% pikseli bez zmian, 5% sól = 255, 5% pieprz = 0). Pokaż oryginał i obraz zaszumiony.

def add_salt_pepper(image):
    noisy_image = np.copy(image)
    mask = np.random.choice([0, 1, 2], size=image.shape[:2], p=[0.9, 0.05, 0.05])
    noisy_image[mask == 1] = 255  # sól
    noisy_image[mask == 2] = 0    # pieprz
    return noisy_image

np.random.seed(42)

chelsea = skimage.data.chelsea()
chelsea_noisy = add_salt_pepper(chelsea)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

axs[0].imshow(chelsea)
axs[0].set_title('Oryginał')
axs[1].imshow(chelsea_noisy)
axs[1].set_title('Szum sól-pieprz')

fig.tight_layout()

# Zadanie 2: Filtr uśredniający vs filtr medianowy
# Odfiltruj zaszumiony obraz jądrem uśredniającym 3×3 (ndimage.correlate) oraz filtrem
# medianowym 3×3 (ndimage.median_filter), osobno dla każdego kanału. Porównaj wyniki.

fig2, axs2 = plt.subplots(1, 3, figsize=(18, 4), sharex=True, sharey=True)

kernel = np.ones((3, 3, 1)) / 9

chelsea_mean = ndimage.correlate(chelsea_noisy, kernel)

chelsea_median = ndimage.median_filter(chelsea_noisy, size=(3, 3, 1))

axs2[0].imshow(chelsea_noisy)
axs2[0].set_title('Szum sól-pieprz')
axs2[1].imshow(chelsea_mean)
axs2[1].set_title('Filtr uśredniający 3×3')
axs2[2].imshow(chelsea_median)
axs2[2].set_title('Filtr medianowy scipy 3×3')

fig2.tight_layout()

# Zadanie 3: Własny filtr medianowy
# Napisz funkcję filtracji medianowej obrazu barwnego (dwie pętle + slicing, mediana osobno
# dla każdego kanału). Pokaż wynik i różnicę względem filtru medianowego z scipy.

def median_self(image, size):
    r = size // 2
    filtered_image = np.copy(image)
    for i in range(r, image.shape[0] - r):
        for j in range(r, image.shape[1] - r):
            window = image[i - r:i + r + 1, j - r:j + r + 1]
            filtered_image[i, j] = np.median(window, axis=(0, 1))
    return filtered_image

fig3, axs3 = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

chelsea_median_own = median_self(chelsea_noisy, 3)

difference = np.abs(chelsea_median.astype(float) - chelsea_median_own.astype(float))
difference = np.sum(difference, axis=2)

axs3[0].imshow(chelsea_median_own)
axs3[0].set_title('Filtr medianowy własny 3×3')
axs3[1].imshow(difference, cmap='gray')
axs3[1].set_title('Różnica scipy vs własny')

fig3.tight_layout()

# Zadanie 4: Obcięta mediana i krawędzie
# Napisz obciętą medianę (przed liczeniem mediany odrzuć z okna wartości 0 i 255).
# Na odszumionym obrazie monochromatycznym wykryj krawędzie operatorem Prewitta i krzyżem Robertsa.

def truncated_median_self(image, size):
    r = size // 2
    filtered_image = np.copy(image)
    for i in range(r, image.shape[0] - r):
        for j in range(r, image.shape[1] - r):
            window = image[i - r:i + r + 1, j - r:j + r + 1]
            values = window[(window > 0) & (window < 255)]
            if values.size > 0:
                filtered_image[i, j] = np.median(values)
    return filtered_image

fig4, axs4 = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)

chelsea_mono = np.mean(chelsea, axis=2)
chelsea_mono_noisy = add_salt_pepper(chelsea_mono)

mono_truncated = truncated_median_self(chelsea_mono_noisy, 3)

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
prewitt_y = prewitt_x.T

roberts_1 = np.array([[1, 0],
                      [0, -1]])
roberts_2 = np.array([[0, 1],
                      [-1, 0]])

prewitt = np.sqrt(ndimage.correlate(mono_truncated, prewitt_x) ** 2 + ndimage.correlate(mono_truncated, prewitt_y) ** 2)
roberts = np.sqrt(ndimage.correlate(mono_truncated, roberts_1) ** 2 + ndimage.correlate(mono_truncated, roberts_2) ** 2)

axs4[0].imshow(prewitt, cmap='gray')
axs4[0].set_title('Prewitt')
axs4[1].imshow(roberts, cmap='gray')
axs4[1].set_title('Krzyż Robertsa')

fig4.tight_layout()

plt.show()