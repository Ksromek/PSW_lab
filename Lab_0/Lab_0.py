import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import negative

#Zadanie_1
mono = np.zeros((30, 30),dtype=int)
print(mono)

mono[10:20, 10:20] = 1
mono[15:25, 5:15] = 2
print(mono)

#Zadanie_2
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

axs[0, 0].imshow(mono)
ax = axs[0, 0]
ax.set_title("obraz monochromatyczny")

axs[0, 1].imshow(mono, cmap='binary')
ax = axs[0, 1]
ax.set_title("obraz monochromatyczny")

fig.tight_layout()
fig.savefig("obraz_mono.png")

#Zadanie_3
color = np.zeros((30, 30,3))
color[15:25, 5:15, 0] = 1
color[10:20, 10:20, 1] = 1
color[5:15, 15:25, 2] = 1

axs[1, 0].imshow(color)
ax = axs[1, 0]
ax.set_title("obraz barwny")

negative=1-color
axs[1, 1].imshow(negative)
ax = axs[1, 1]
ax.set_title("negatyw")
plt.show()

fig.tight_layout()
fig.savefig("obraz_all.png")

