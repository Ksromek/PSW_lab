import numpy as np
import matplotlib.pyplot as plt

#Zadanie_1
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

x=np.linspace(0,4*np.pi,40)

y=np.sin(x)

fig.tight_layout()

axs[0,0].plot(x,y,color='red')
axs[0, 0].set_title("Funkcja sinus")

y=y[:, np.newaxis]*y[np.newaxis, :]

axs[0, 1].imshow(y, cmap='magma')

axs[0, 1].set_title(f"min: {np.min(y):.3f}, max: {np.max(y):.3f}")

y_norm=y-np.min(y)

y_norm=y_norm/np.max(y_norm)

axs[0, 2].imshow(y_norm, cmap='magma')

axs[0, 2].set_title(f"min: {np.min(y_norm):.3f}, max: {np.max(y_norm):.3f}")

fig.savefig("Zad_1.png")

#Zadanie_2

y_discr_2=y_norm*(2 ** 2 -1)
y_discr_2 = np.rint(y_discr_2)

axs[1, 0].imshow(y_discr_2, cmap='magma')
axs[1, 0].set_title(f"min: {np.min(y_discr_2):.0f}, max: {np.max(y_discr_2):.0f}")

y_discr_4=y_norm*(2 ** 4 -1)
y_discr_4 = np.rint(y_discr_4)

axs[1, 1].imshow(y_discr_4, cmap='magma')
axs[1, 1].set_title(f"min: {np.min(y_discr_4):.0f}, max: {np.max(y_discr_4):.0f}")

y_discr_8=y_norm*(2 ** 8 -1)
y_discr_8 = np.rint(y_discr_8)

axs[1, 2].imshow(y_discr_4, cmap='magma')
axs[1, 2].set_title(f"min: {np.min(y_discr_8):.0f}, max: {np.max(y_discr_8):.0f}")

fig.savefig("Zad_2.png")

#Zadanie_3

gaussian_noise=np.random.normal(size=(40, 40))

y_gaussian_noise=y_norm+gaussian_noise

axs[2, 0].imshow(y_gaussian_noise, cmap='magma')
axs[2, 0].set_title("n=1")

y_noise_50=y_norm+np.random.normal(size=(50,40,40))
y_denoised_50=np.mean(y_noise_50, axis=0)

axs[2, 1].imshow(y_denoised_50, cmap='magma')
axs[2, 1].set_title("n=50")

y_noise_1000=y_norm+np.random.normal(size=(1000,40,40))
y_denoised_1000=np.mean(y_noise_1000, axis=0)

axs[2, 2].imshow(y_denoised_1000, cmap='magma')
axs[2, 2].set_title("n=1000")

fig.savefig("Zad_3.png")

plt.show()
