import numpy as np
import matplotlib.pyplot as plt
import skimage


# Zadanie 1: Transformata i transformata odwrotna
# Dla obrazu camera policz fft2. Pokaż część rzeczywistą, urojoną, amplitudę (log) i fazę (arctan2).
# Wróć do obrazu przez ifftshift + ifft2 i pokaż różnicę z oryginałem.

camera = skimage.data.camera().astype(float)

fft_camera = np.fft.fft2(camera)
fft_camera_shifted = np.fft.fftshift(fft_camera)

magnitude = np.log(np.abs(fft_camera_shifted) + 1)
phase = np.arctan2(fft_camera_shifted.imag, fft_camera_shifted.real)

camera_back = np.fft.ifft2(np.fft.ifftshift(fft_camera_shifted)).real

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(camera, cmap='gray')
axs[0, 0].set_title('oryginał')
axs[0, 1].imshow(np.log(np.abs(fft_camera_shifted.real) + 1), cmap='gray')
axs[0, 1].set_title('część rzeczywista (log |real|)')
axs[0, 2].imshow(np.log(np.abs(fft_camera_shifted.imag) + 1), cmap='gray')
axs[0, 2].set_title('część urojona (log |imag|)')
axs[1, 0].imshow(magnitude, cmap='gray')
axs[1, 0].set_title('amplituda (log |F|)')
axs[1, 1].imshow(phase, cmap='gray')
axs[1, 1].set_title('faza (arctan2)')
axs[1, 2].imshow(np.abs(camera - camera_back), cmap='gray')
axs[1, 2].set_title(f'|oryginał − ifft|, max = {np.max(np.abs(camera - camera_back)):.1e}')

fig.tight_layout()
fig.savefig("Zad_1.png")

# Zadanie 2: Modyfikacja widma slicingiem
# W przesuniętym widmie zostaw tylko kwadrat 60×60 wokół środka (dolnoprzepustowy),
# albo wyzeruj ten kwadrat (górnoprzepustowy). Wróć do obrazu przez ifftshift + ifft2.

center = camera.shape[0] // 2
r = 30

low_pass = np.zeros_like(fft_camera_shifted)
low_pass[center - r:center + r, center - r:center + r] = fft_camera_shifted[center - r:center + r, center - r:center + r]

high_pass = np.copy(fft_camera_shifted)
high_pass[center - r:center + r, center - r:center + r] = 0

camera_low = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
camera_high = np.fft.ifft2(np.fft.ifftshift(high_pass)).real

fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))

axs2[0, 0].imshow(np.log(np.abs(low_pass) + 1), cmap='gray')
axs2[0, 0].set_title('widmo: tylko środek')
axs2[0, 1].imshow(camera_low, cmap='gray')
axs2[0, 1].set_title('dolnoprzepustowy (rozmycie)')
axs2[1, 0].imshow(np.log(np.abs(high_pass) + 1), cmap='gray')
axs2[1, 0].set_title('widmo: bez środka')
axs2[1, 1].imshow(camera_high, cmap='gray')
axs2[1, 1].set_title('górnoprzepustowy (krawędzie)')

fig2.tight_layout()
fig2.savefig("Zad_2.png")

# Zadanie 3: Filtry idealny, gaussowski i Butterwortha
# Zbuduj maski dolnoprzepustowe o promieniu D0 = 30: idealną, gaussowską i Butterwortha (n = 2).
# Pomnóż przez nie widmo camera, wróć do obrazu i porównaj (efekt dzwonienia filtru idealnego).
# W ostatnim wierszu filtr górnoprzepustowy gaussowski (1 − maska).

H, W = camera.shape
row_idx = np.arange(H)[:, np.newaxis]
col_idx = np.arange(W)[np.newaxis, :]
D = np.sqrt((row_idx - H // 2) ** 2 + (col_idx - W // 2) ** 2)   # odległość od środka widma

D0 = 30
ideal = (D <= D0).astype(float)
gauss = np.exp(-D ** 2 / (2 * D0 ** 2))
butter = 1 / (1 + (D / D0) ** (2 * 2))

masks = [ideal, gauss, butter]
names = ['idealny', 'gaussowski', 'Butterworth (n = 2)']

fig3, axs3 = plt.subplots(3, 3, figsize=(15, 15))

for k in range(3):
    filtered = np.fft.ifft2(np.fft.ifftshift(fft_camera_shifted * masks[k])).real
    axs3[0, k].imshow(masks[k], cmap='gray')
    axs3[0, k].set_title(f'maska: {names[k]}')
    axs3[1, k].imshow(filtered, cmap='gray')
    axs3[1, k].set_title(f'dolnoprzepustowy: {names[k]}')

high_gauss = 1 - gauss
camera_high_gauss = np.fft.ifft2(np.fft.ifftshift(fft_camera_shifted * high_gauss)).real

axs3[2, 0].imshow(high_gauss, cmap='gray')
axs3[2, 0].set_title('maska górnoprzepustowa (1 − gauss)')
axs3[2, 1].imshow(camera_high_gauss, cmap='gray')
axs3[2, 1].set_title('górnoprzepustowy: krawędzie')
axs3[2, 2].imshow(np.log(np.abs(fft_camera_shifted * high_gauss) + 1), cmap='gray')
axs3[2, 2].set_title('widmo po filtracji')

fig3.tight_layout()
fig3.savefig("Zad_3.png")

# Zadanie 4: Usuwanie szumu okresowego (notch filter)
# Dodaj do camera szum okresowy (ukośne prążki). W widmie pojawią się dwa jasne punkty.
# Wyzeruj je małymi kółkami (filtr wycinający) i wróć do obrazu.

noise = 40 * np.sin(2 * np.pi * (40 * col_idx / W + 30 * row_idx / H))   # 40 okresów w poziomie, 30 w pionie
camera_striped = camera + noise

fft_striped = np.fft.fftshift(np.fft.fft2(camera_striped))

magnitude_striped = np.abs(fft_striped)
magnitude_no_center = np.copy(magnitude_striped)
magnitude_no_center[D < 10] = 0                                    # pomijamy niskie częstotliwości obrazu
peaks = np.argwhere(magnitude_no_center >= np.sort(magnitude_no_center.flatten())[-2])

notch = np.ones((H, W))
for pr, pc in peaks:
    dist = np.sqrt((row_idx - pr) ** 2 + (col_idx - pc) ** 2)   # odległość od wykrytego punktu szumu
    notch[dist <= 5] = 0

camera_cleaned = np.fft.ifft2(np.fft.ifftshift(fft_striped * notch)).real

fig4, axs4 = plt.subplots(2, 2, figsize=(10, 10))

axs4[0, 0].imshow(camera_striped, cmap='gray')
axs4[0, 0].set_title('szum okresowy')
axs4[0, 1].imshow(np.log(magnitude_striped + 1), cmap='gray')
axs4[0, 1].set_title('widmo (dwa jasne punkty = szum)')
axs4[1, 0].imshow(np.log(np.abs(fft_striped * notch) + 1), cmap='gray')
axs4[1, 0].set_title('widmo po wycięciu')
axs4[1, 1].imshow(camera_cleaned, cmap='gray')
axs4[1, 1].set_title('po filtracji notch')

fig4.tight_layout()
fig4.savefig("Zad_4.png")

# Zadanie 5: Amplituda czy faza?
# Weź obrazy camera i moon (oba 512×512). Złóż dwa obrazy: amplituda camera + faza moon
# i odwrotnie (F = |F| · e^(i·faza)). Który obraz „widać” w wyniku?

moon = skimage.data.moon().astype(float)
fft_moon = np.fft.fft2(moon)

cam_amp_moon_phase = np.fft.ifft2(np.abs(fft_camera) * np.exp(1j * np.angle(fft_moon))).real
moon_amp_cam_phase = np.fft.ifft2(np.abs(fft_moon) * np.exp(1j * np.angle(fft_camera))).real

fig5, axs5 = plt.subplots(2, 2, figsize=(10, 10))

axs5[0, 0].imshow(camera, cmap='gray')
axs5[0, 0].set_title('camera')
axs5[0, 1].imshow(moon, cmap='gray')
axs5[0, 1].set_title('moon')
axs5[1, 0].imshow(cam_amp_moon_phase, cmap='gray')
axs5[1, 0].set_title('amplituda camera + faza moon')
axs5[1, 1].imshow(moon_amp_cam_phase, cmap='gray')
axs5[1, 1].set_title('amplituda moon + faza camera')

fig5.tight_layout()
fig5.savefig("Zad_5.png")

plt.show()