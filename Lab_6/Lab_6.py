import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk

def show_fft(axs, image):
    fft_image = np.fft.fftshift(np.fft.fft2(image))

    axs[0, 0].imshow(image, cmap='magma')
    axs[0, 0].set_title('original')

    axs[0, 1].imshow(np.log(np.abs(fft_image.real) + 1), cmap='magma')
    axs[0, 1].set_title('real')

    axs[0, 2].imshow(np.log(np.abs(fft_image.imag) + 1), cmap='magma')
    axs[0, 2].set_title('imag')

    phase = np.arctan2(fft_image.imag, fft_image.real)
    axs[1, 0].imshow(phase, cmap='magma')
    axs[1, 0].set_title('phase')

    magnitude = np.abs(fft_image)
    axs[1, 1].imshow(np.log(magnitude + 1), cmap='magma')
    axs[1, 1].set_title('magnitude')

    inverse = np.fft.ifft2(np.fft.ifftshift(fft_image)).real
    axs[1, 2].imshow(inverse, cmap='magma')
    axs[1, 2].set_title('inverse')


#Zadanie_1:
image = np.zeros((1000, 1000))
image[500:520, 460:550] = 1

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

show_fft(axs, image)

fig.tight_layout()
fig.savefig("Zad_1.png")

#Zadanie_2:
x, y = np.meshgrid(np.linspace(0, 15, 1000), np.linspace(0, 15, 1000))

amplitudes = [2, 4, 6, 8, 10]
angles = [1,3,5,7,9]
wavelengths = [3, 5, 7, 3, 9]

sines = np.zeros((1000, 1000))
for amplitude, angle, wavelength in zip(amplitudes, angles, wavelengths):
    sines = sines + amplitude * np.sin(2 * np.pi * (x * np.cos(angle) + y * np.sin(angle)) * (1 / wavelength))

sines = sines - np.min(sines)
sines = sines / np.max(sines)

fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
show_fft(axs2, sines)

fig2.tight_layout()
fig2.savefig("Zad_2.png")

#Zadanie_3:
def normalize(channel):
    channel = channel - np.min(channel)
    return channel / np.max(channel)

fig3, axs3 = plt.subplots(2, 3, figsize=(15, 7))

badger = plt.imread('badger.jpg')
badger_mono = np.mean(badger[:, :, :3], axis=2)

fft_badger = np.fft.fftshift(np.fft.fft2(badger_mono))

mask = np.zeros(badger_mono.shape)
center = (badger_mono.shape[0] // 2, badger_mono.shape[1] // 2)
radius = min(badger_mono.shape) // 10
rr, cc = disk(center, radius, shape=badger_mono.shape)
mask[rr, cc] = 1

fft_badger = np.fft.ifftshift(fft_badger * mask)

red = normalize(np.fft.ifft2(fft_badger.real).real)
green = normalize(np.fft.ifft2(1j * fft_badger.imag).real)
blue = normalize(np.fft.ifft2(fft_badger).real)

badger_color = np.stack([red, green, blue], axis=2)

axs3[0, 0].imshow(badger_mono, cmap='gray')
axs3[0, 1].imshow(np.log(np.abs(np.fft.fftshift(fft_badger)) + 1))
axs3[0, 2].imshow(badger_color)

axs3[1, 0].imshow(red, cmap='Reds_r')
axs3[1, 1].imshow(green, cmap='Greens_r')
axs3[1, 2].imshow(blue, cmap='Blues_r')

fig3.tight_layout()
fig3.savefig("Zad_3.png")

plt.show()
