import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.spatial.distance import cdist
from skimage.transform import AffineTransform, warp


def make_grid(row_values, col_values):
    grid = []
    for w in row_values:
        for k in col_values:
            grid.append([w, k])
    return np.array(grid)

# Zadanie 1:

fig, axs = plt.subplots(2, 2, figsize=(10, 7))

cat = skimage.data.cat()

axs[0, 0].imshow(cat)
axs[0, 0].set_title('Chelsea')

cat_mono = np.mean(cat, axis=2)
cat_reduced_mono = cat_mono[::8, ::8]
axs[0, 1].imshow(cat_reduced_mono, cmap='binary_r')
axs[0, 1].set_title("Zmniejszenie + mono")

angle = np.deg2rad(15)

rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,             1]
])

shear_matrix = np.array([
    [1, 0.5, 0],
    [0, 1,   0],
    [0, 0,   1]
])

rotation = AffineTransform(matrix=rotation_matrix)
cat_rotated = warp(cat_reduced_mono, rotation.inverse)
axs[1, 0].imshow(cat_rotated, cmap='binary_r')
axs[1, 0].set_title('Rotacja')

leaning = AffineTransform(matrix=shear_matrix)
cat_leaned = warp(cat_reduced_mono, leaning.inverse)
axs[1, 1].imshow(cat_leaned, cmap='binary_r')
axs[1, 1].set_title('Pochylenie')

fig.tight_layout()

fig.savefig("Zad_1.png")

#Zadanie_2:

fig2, axs2 = plt.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)

coords = make_grid(range(cat_reduced_mono.shape[0]), range(cat_reduced_mono.shape[1]))

x = coords[:, 0]
y = coords[:, 1]

intensity = cat_reduced_mono.flatten()

axs2[0].scatter(x, y, c=intensity, cmap='binary_r')
axs2[0].set_title('Oryginał')

ones = np.ones(coords.shape[0])

coords_extended = np.column_stack((coords, ones))

rotation_matrix_2 = rotation_matrix

new_coords = (coords_extended @ rotation_matrix_2)

x_new = new_coords[:, 0]
y_new = new_coords[:, 1]

axs2[1].scatter(x_new, y_new, c=intensity, cmap='binary_r')
axs2[1].set_title('Rotacja (macierzowo)')

fig.tight_layout()

fig2.savefig("Zad_2.png")

## Zadanie_3:

fig3, axs3 = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)

axs3[0].scatter(x, y, c=intensity, cmap='binary_r')
axs3[0].set_title('Oryginał')

np.random.seed(42)
indexes = np.random.choice(coords.shape[0], 1000, replace=False)

x_rand = x[indexes]
y_rand = y[indexes]
intensity_rand = intensity[indexes]

axs3[1].scatter(x_rand, y_rand, c=intensity_rand, cmap='binary_r')
axs3[1].set_title('Sygnał wybrakowany - 1000 losowych punktów')

source_cat = np.column_stack((x_rand, y_rand))

x_interpolation = np.linspace(0, cat_reduced_mono.shape[0] - 1, 300)
y_interpolation = np.linspace(0, cat_reduced_mono.shape[1] - 1, 400)

coords_interpolation = make_grid(x_interpolation, y_interpolation)

distance = cdist(coords_interpolation, source_cat)

nearest_neighbour_index = np.argmin(distance, axis=1)

values_interpolated = intensity_rand[nearest_neighbour_index]

axs3[2].scatter(coords_interpolation[:, 0], coords_interpolation[:, 1], c=values_interpolated, cmap='binary_r')
axs3[2].set_title('Interpolacja - Najbliższy sąsiad (300×400)')

fig.tight_layout()

fig3.savefig("Zad_3.png")

plt.show()