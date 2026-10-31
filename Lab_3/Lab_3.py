import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.stats import entropy

#Zadanie_1:
fig, axs = plt.subplots(6, 3, figsize=(12, 18))

cat = skimage.data.cat()

L = pow(2, 8)

identity_transform = np.linspace(0, L - 1, L).astype(int)

negation = np.linspace(L - 1, 0, L).astype(int)

prog_function = []
for i in range(L):
    if i < 50 or i > 150:
        prog_function.append(0)
    else:
        prog_function.append(L - 1)
prog_function = np.array(prog_function)

sinus = np.sin(np.linspace(0, 2 * np.pi, L))
sinus_norm = sinus - np.min(sinus)
sinus_norm = sinus_norm / np.max(sinus_norm)
sinus = np.rint(sinus_norm * (L - 1)).astype(int)

gamma_correction_0_3 = pow(identity_transform, 0.3)
gamma_0_3_norm = gamma_correction_0_3 - np.min(gamma_correction_0_3)
gamma_0_3_norm = gamma_0_3_norm / np.max(gamma_0_3_norm)
gamma_correction_0_3 = np.rint(gamma_0_3_norm * (L - 1)).astype(int)

gamma_correction_3 = pow(identity_transform, 3)
gamma_3_norm = gamma_correction_3 - np.min(gamma_correction_3)
gamma_3_norm = gamma_3_norm / np.max(gamma_3_norm)
gamma_correction_3 = np.rint(gamma_3_norm * (L - 1)).astype(int)

x_lut = np.linspace(0, L - 1, L)

axs[0, 0].scatter(x_lut, identity_transform, c='black', s=1)

axs[1, 0].scatter(x_lut, negation, c='black', s=1)

axs[2, 0].scatter(x_lut, prog_function, c='black', s=1)

axs[3, 0].scatter(x_lut, sinus, c='black', s=1)

axs[4, 0].scatter(x_lut, gamma_correction_0_3, c='black', s=1)

axs[5, 0].scatter(x_lut, gamma_correction_3, c='black', s=1)


axs[0, 1].imshow(identity_transform[cat])

axs[1, 1].imshow(negation[cat])

axs[2, 1].imshow(prog_function[cat])

axs[3, 1].imshow(sinus[cat])

axs[4, 1].imshow(gamma_correction_0_3[cat])

axs[5, 1].imshow(gamma_correction_3[cat])

def calculate_histogram(img, x, y):
    r_values, r_counts = np.unique(img[:, :, 0], return_counts=True)
    r_prob = r_counts / np.sum(r_counts)
    axs[x, y].scatter(r_values, r_prob, c='r', s=1)

    g_values, g_counts = np.unique(img[:, :, 1], return_counts=True)
    g_prob = g_counts / np.sum(g_counts)
    axs[x, y].scatter(g_values, g_prob, c='g', s=1)

    b_values, b_counts = np.unique(img[:, :, 2], return_counts=True)
    b_prob = b_counts / np.sum(b_counts)
    axs[x, y].scatter(b_values, b_prob, c='b', s=1)

    axs[x, y].set_xlim(0, L - 1)

calculate_histogram(identity_transform[cat], 0, 2)
calculate_histogram(negation[cat], 1, 2)
calculate_histogram(prog_function[cat], 2, 2)
calculate_histogram(sinus[cat], 3, 2)
calculate_histogram(gamma_correction_0_3[cat], 4, 2)
calculate_histogram(gamma_correction_3[cat], 5, 2)

fig.tight_layout()

fig.savefig("Zad_1.png")

#Zadanie_2:

fig2, axs2 = plt.subplots(2, 3, figsize=(10, 7))

moon = skimage.data.moon()

axs2[0, 0].imshow(moon, cmap='binary_r')
axs2[0, 0].set_title('Obraz oryginalny')

moon_values = np.arange(L)
moon_counts = np.bincount(moon.flatten(), minlength=L)
moon_prob = moon_counts / np.sum(moon_counts)

axs2[0, 1].bar(moon_values, moon_prob, color='black', width=1)
axs2[0, 1].set_title('Histogram')
axs2[0, 1].set_xlim(0, L - 1)

distributant_moon = np.cumsum(moon_prob)

axs2[0, 2].scatter(moon_values, distributant_moon, c='black', s=1)
axs2[0, 2].set_title('Dystrybuanta')
axs2[0, 2].set_xlim(0, L - 1)

lut_moon = np.rint(distributant_moon * (L - 1)).astype(int)

axs2[1, 0].scatter(moon_values, lut_moon, c='black', s=1)
axs2[1, 0].set_title('LUT')
axs2[1, 0].set_xlim(0, L - 1)

lut_transformed_moon = lut_moon[moon]

axs2[1, 1].imshow(lut_transformed_moon, cmap='binary_r')
axs2[1, 1].set_title('Obraz po wyrównaniu')

transformed_moon_counts = np.bincount(lut_transformed_moon.flatten(), minlength=L)
transformed_moon_prob = transformed_moon_counts / np.sum(transformed_moon_counts)

axs2[1, 2].bar(moon_values, transformed_moon_prob, color='black', width=1)
axs2[1, 2].set_title('Histogram po wyrównaniu')
axs2[1, 2].set_xlim(0, L - 1)

fig2.tight_layout()

fig2.savefig("Zad_2.png")

#Zadanie_3:

fig3, axs3 = plt.subplots(1, 3, figsize=(15, 5))

intensity_color_moon = np.zeros((10, 10, 10))

sampling_values = np.linspace(0, 7 * np.pi, 10)

def make_sin_lut(start, end):
    sin_lut = np.sin(np.linspace(start, end, L))
    sin_lut_norm = sin_lut - np.min(sin_lut)
    sin_lut_norm = sin_lut_norm / np.max(sin_lut_norm)
    sin_lut = np.rint(sin_lut_norm * (L - 1)).astype(int)
    return sin_lut

red_luts = []
green_luts = []
blue_luts = []

for end in sampling_values:
    red_luts.append(make_sin_lut(-0.5 * np.pi, end))
    green_luts.append(make_sin_lut(-0.75 * np.pi, end))
    blue_luts.append(make_sin_lut(-np.pi, end))


for i in range(10):
    for j in range(10):
        for k in range(10):
            color_moon = np.stack([red_luts[i][moon],
                                   green_luts[j][moon],
                                   blue_luts[k][moon]], axis=2)

            counts = np.bincount(color_moon.flatten(), minlength=L)
            prob = counts / np.sum(counts)

            intensity_color_moon[i, j, k] = entropy(prob)

best_i, best_j, best_k = np.argwhere(intensity_color_moon == np.max(intensity_color_moon))[0]

best_moon = np.stack([red_luts[best_i][moon],
                      green_luts[best_j][moon],
                      blue_luts[best_k][moon]], axis=2)

x_lut = np.arange(L)
axs3[0].scatter(x_lut, red_luts[best_i], c='r', s=1)
axs3[0].scatter(x_lut, green_luts[best_j], c='g', s=1)
axs3[0].scatter(x_lut, blue_luts[best_k], c='b', s=1)
axs3[0].set_title('Funkcje transformacji')

axs3[1].imshow(best_moon)
axs3[1].set_title('Obraz po transformacji')


for channel, color in enumerate(['r', 'g', 'b']):
    counts = np.bincount(best_moon[:, :, channel].flatten(), minlength=L)
    prob = counts / np.sum(counts)
    axs3[2].bar(x_lut, prob, color=color, width=1)

axs3[2].set_title('Histogramy')
axs3[2].set_xlim(0, L - 1)

fig3.tight_layout()
fig3.savefig("Zad_3.png")
plt.show()
