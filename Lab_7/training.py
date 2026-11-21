import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def normalize(channel):
    channel = channel - np.min(channel)
    return channel / np.max(channel)


np.random.seed(42)

# Zadanie 1: Obraz nadwidmowy i jego pasma
# Wczytaj Indian Pines (obraz i ground truth). Pokaż 5 losowo wybranych pasm (np.random.choice)
# oraz mapę ground truth z colormapą turbo.

cube = loadmat('Indian_pines_corrected.mat')['indian_pines_corrected'].astype(float)   # (145, 145, 200)
gt = loadmat('Indian_pines_gt.mat')['indian_pines_gt']                                # (145, 145), klasy 0–16

H, W, B = cube.shape
print('kształt obrazu:', cube.shape, '| klasy:', np.unique(gt))

bands = np.sort(np.random.choice(B, 5, replace=False))

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()                     # siatka 2×3 jako lista 6 osi

for k, band in enumerate(bands):
    axs[k].imshow(cube[:, :, band], cmap='gray')
    axs[k].set_title(f'pasmo {band}')

axs[5].imshow(gt, cmap='turbo')
axs[5].set_title('ground truth')

fig.tight_layout()
fig.savefig("Zad_1.png")

# Zadanie 2: Sygnatury spektralne
# W pierwszej komórce: sygnatury spektralne 5 losowych pikseli (wartości we wszystkich pasmach).
# W drugiej: średnia sygnatura każdej klasy z ground truth (bez tła, klasa 0), kolory z plt.cm.turbo.

fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5))

for _ in range(5):
    i = np.random.choice(H)
    j = np.random.choice(W)
    axs2[0].plot(cube[i, j, :], label=f'piksel ({i}, {j})')
axs2[0].set_title('sygnatury losowych pikseli')
axs2[0].set_xlabel('numer pasma')
axs2[0].legend()

classes = np.unique(gt)[1:]             # bez klasy 0 (tło, brak etykiety)
for c in classes:
    mean_signature = np.mean(cube[gt == c], axis=0)   # cube[gt == c] → (liczba pikseli klasy, 200)
    axs2[1].plot(mean_signature, color=plt.cm.turbo(c / classes.max()), label=f'klasa {c}')
axs2[1].set_title('średnie sygnatury klas')
axs2[1].set_xlabel('numer pasma')
axs2[1].legend(fontsize=7, ncol=2)

fig2.tight_layout()
fig2.savefig("Zad_2.png")

# Zadanie 3: Ekstrakcja cech PCA
# Spłaszcz obraz do macierzy (piksele × pasma), wykonaj PCA. Pokaż 3 pierwsze składowe jako obrazy,
# obraz barwny z 3 składowych (każda znormalizowana do 0–1) i skumulowaną wariancję wyjaśnioną.

X = cube.reshape(-1, B)                                  # (145*145, 200): każdy wiersz to sygnatura jednego piksela
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)         # standaryzacja każdego pasma

pca = PCA()
X_pca = pca.fit_transform(X)                             # (145*145, 200): piksele w nowych współrzędnych
components = X_pca.reshape(H, W, B)                      # z powrotem do kształtu obrazu

fig3, axs3 = plt.subplots(2, 3, figsize=(12, 8))

for k in range(3):
    axs3[0, k].imshow(components[:, :, k], cmap='gray')
    axs3[0, k].set_title(f'PC{k + 1} ({pca.explained_variance_ratio_[k] * 100:.1f}% wariancji)')

pca_rgb = np.stack([normalize(components[:, :, k]) for k in range(3)], axis=2)
axs3[1, 0].imshow(pca_rgb)
axs3[1, 0].set_title('PC1–PC3 jako RGB')

axs3[1, 1].imshow(gt, cmap='turbo')
axs3[1, 1].set_title('ground truth')

cumulative = np.cumsum(pca.explained_variance_ratio_)
axs3[1, 2].plot(np.arange(1, B + 1), cumulative)
axs3[1, 2].axhline(0.95, color='r', ls=':')
axs3[1, 2].set_xlim(0, 30)
axs3[1, 2].set_title(f'wariancja skumulowana (95%: {np.argmax(cumulative >= 0.95) + 1} PC)')
axs3[1, 2].grid(ls=':')

fig3.tight_layout()
fig3.savefig("Zad_3.png")

# Zadanie 4: Klasyfikacja z ground truth
# Weź tylko piksele z etykietą (gt > 0). Podziel na zbiór uczący i testowy (stratyfikacja),
# naucz klasyfikator k-NN na 200 pasmach i na kilku składowych PCA. Porównaj dokładność.
# Pokaż mapę predykcji obok ground truth.

labels = gt.flatten()
labeled = labels > 0

for n_components in [200, 30, 10, 3]:
    features = X_pca[:, :n_components] if n_components < B else X
    X_train, X_test, y_train, y_test = train_test_split(features[labeled], labels[labeled],
                                                        test_size=0.3, stratify=labels[labeled],
                                                        random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    print(f'cechy: {n_components:3d} | dokładność: {accuracy_score(y_test, knn.predict(X_test)):.3f}')

X_train, X_test, y_train, y_test = train_test_split(X_pca[labeled, :10], labels[labeled],
                                                    test_size=0.3, stratify=labels[labeled], random_state=42)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)    # uczony tylko na zbiorze uczącym

prediction = np.zeros(H * W, dtype=int)                 # tło (klasa 0) zostaje zerem
prediction[labeled] = knn.predict(X_pca[labeled, :10])

fig4, axs4 = plt.subplots(1, 2, figsize=(10, 5))
axs4[0].imshow(gt, cmap='turbo')
axs4[0].set_title('ground truth')
axs4[1].imshow(prediction.reshape(H, W), cmap='turbo')
axs4[1].set_title('predykcja k-NN (10 składowych PCA)')

fig4.tight_layout()
fig4.savefig("Zad_4.png")

# Zadanie 5 (dodatkowe): PCA ręcznie
# Policz macierz kowariancji, jej wektory i wartości własne (np.linalg.eigh), posortuj malejąco
# i zrzutuj dane. Porównaj z wynikiem sklearn (składowe mogą mieć odwrotny znak).

covariance = np.cov(X, rowvar=False)                    # (200, 200)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # eigh – dla macierzy symetrycznej, wynik rosnąco
order = np.argsort(eigenvalues)[::-1]                   # indeksy od największej wartości własnej
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

X_manual = X @ eigenvectors[:, :3]                      # rzut na 3 pierwsze kierunki

print('wariancja wyjaśniona (ręcznie):', np.round(eigenvalues[:3] / np.sum(eigenvalues), 4))
print('wariancja wyjaśniona (sklearn):', np.round(pca.explained_variance_ratio_[:3], 4))
print('zgodność składowych (z dokładnością do znaku):',
      np.allclose(np.abs(X_manual), np.abs(X_pca[:, :3]), atol=1e-6))

plt.show()