## 1. Declaring arrays

```python
array = np.array([1, 2, 3])                    # array from a list
zero_array = np.zeros((n, m))                  # n × m array of zeros (shape as a tuple!)
zero_array_with_type = np.zeros((n, m), dtype=int)  # same, with integer type
ones_array = np.ones(n)                        # vector of n ones
array_copy = np.copy(array)                    # independent copy (changes don't affect the original)
```

### Ranges of values
```python
x = np.linspace(m, n, k)    # m – start, n – stop (included), k – number of samples
x = np.arange(m, n, s)      # m – start, n – stop (excluded), s – step (default 1)
```

### Random arrays
```python
img = np.random.choice(256, size=(m, n))              # random integers from 0 to 255, shape m × n
idx = np.random.choice(n, k, replace=False)           # k different random indices from 0..n-1
np.random.seed(42)                                    # same random numbers on every run
```

### Grid of coordinates
```python
rr, cc = np.meshgrid(rows, cols, indexing='ij')                # two 2D arrays: row and column of each pixel
coords = np.stack([rr.flatten(), cc.flatten()], axis=1)        # (N, 2) array of [row, col] pairs
```

---

## 2. Filling arrays

```python
array[m:n, k:l] = x    # m:n – row range, k:l – column range
```

Ranges are half-open: `m:n` covers indices `m, m+1, …, n-1`.

### Slicing variants (1D)

| Slice      | Meaning                                  |
|------------|------------------------------------------|
| `a[k]`     | single element at index `k`              |
| `a[m:n]`   | from `m` to `n-1`                        |
| `a[:n]`    | from start to `n-1` (first `n` elements) |
| `a[k:]`    | from `k` to the end                      |
| `a[:]`     | whole array                              |
| `a[-1]`    | last element                             |
| `a[-k:]`   | last `k` elements                        |
| `a[:-k]`   | everything except last `k` elements      |
| `a[::s]`   | every `s`-th element                     |
| `a[m:n:s]` | from `m` to `n-1` with step `s`          |
| `a[k::s]`  | from `k` to the end with step `s`        |
| `a[::-1]`  | reversed array                           |

### Slicing variants (2D)

| Slice          | Meaning                                    |
|----------------|--------------------------------------------|
| `a[i, j]`      | single element (row `i`, column `j`)       |
| `a[i, :]`      | whole row `i` (same as `a[i]`)             |
| `a[:, j]`      | whole column `j`                           |
| `a[m:n, :]`    | rows `m` to `n-1`, all columns             |
| `a[:, k:l]`    | all rows, columns `k` to `l-1`             |
| `a[::2, ::2]`  | every 2nd row and every 2nd column         |
| `a[::-1, :]`   | rows in reversed order (flip vertically)   |
| `a[:, ::-1]`   | columns in reversed order (flip horizontally) |

```python
a[:, :] = 0            # set whole array to 0
a[:5, :] = 1           # first 5 rows set to 1
a[:, -3:] = 2          # last 3 columns set to 2
a[::2, ::2] = 3        # checkerboard-like pattern (every 2nd element)
```

### Boolean masks
```python
a[a > 0] = np.inf      # set every element greater than 0 to infinity
mask = a < 0           # array of True/False with the same shape as a
```

### Indexing with an array of indices
```python
idx = np.array([0, 0, 1, 2, 2])
b = a[idx]             # picks a[0], a[0], a[1], a[2], a[2]
```

### Min / max position
```python
i = np.argmin(a)             # index of the smallest element
i = np.argmax(a)             # index of the largest element
idx = np.argmin(d, axis=0)   # index of the minimum in each column
idx = np.argmin(d, axis=1)   # index of the minimum in each row
pos = np.argwhere(a == np.max(a))[0]   # coordinates (e.g. [i, j, k]) of the maximum in an N-D array
```

### Reshaping and joining
```python
v = a.flatten()                          # any array → 1D vector (row by row)
t = a.T                                  # transpose (rows ↔ columns)
c = np.column_stack((a, b))              # glue vectors/arrays side by side as columns
s = np.stack([r, g, b], axis=2)          # three (H, W) arrays → one (H, W, 3) image
```

---

## 3. Figures and axes

```python
fig, axs = plt.subplots(m, n, figsize=(x, y))  # m – rows, n – columns; figsize in inches (width, height)

axs[0, 0].imshow(x, cmap='magma')   # x – what we want to show, cmap – colour map
axs[0, 0].set_title("Title")

fig.tight_layout()                  # fix spacing between subplots (call after drawing!)
fig.savefig("figure.png")           # save to file
```

With one row or one column (`subplots(1, n)`) `axs` is **1D** → use `axs[0]`, `axs[1]`, not `axs[0, 0]`.

```python
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)   # subplots share the same axis ranges
fig.savefig("figure.png", dpi=200)                        # higher resolution
```

### Other plot types and options
```python
plt.imshow(img, cmap='binary_r')                 # grayscale: 0 – black, max – white
plt.imshow(img, cmap='binary_r', vmin=0, vmax=255)   # fixed colour range
plt.scatter(x, y, color='tomato')                # points instead of a line
plt.scatter(x, y, marker='x', color='b')         # custom marker and colour
plt.scatter(x, y, c=values, cmap='binary_r', s=1)    # colour of each point from a vector, s – point size
plt.bar(x, heights, color='black', width=1, alpha=0.5)  # bar chart; alpha – transparency
plt.grid(ls=':')                                 # dotted grid
```

### Axis settings
```python
ax.set_xlim(0, 255)       # fixed x range
ax.set_aspect('equal')    # same scale on x and y (no stretching)
ax.invert_yaxis()         # y grows downwards (like in images)
```

---

## 4. Signal operations

### 1D → 2D
```python
y = y[:, np.newaxis] * y[np.newaxis, :]   # outer product: 1D array → 2D array
```

### Normalization (to range [0, 1])
```python
y_norm = y - np.min(y)
y_norm = y_norm / np.max(y_norm)
```

### Discretization (n bits)
```python
y_discr_n = y_norm * (2 ** n - 1)   # scale to [0, 2^n - 1]
y_rint = np.rint(y_discr_n)         # round to the nearest integer
y_int = y_rint.astype(int)          # change type to int
```

`.astype(int)` alone **cuts off** the fraction (12.7 → 12), so round with `np.rint` first.

### Gaussian noise
```python
gaussian_noise = np.random.normal(size=(m, n))    # noise of a given shape
gaussian_noise = np.random.normal(size=y.shape)   # noise with the same shape as y
```

### Adding noise and denoising
```python
y_noise_n = y + np.random.normal(size=(n, 40, 40))   # n noisy copies of a 40×40 image
y_denoised_n = np.mean(y_noise_n, axis=0)            # averaging copies → denoised image
```

---

## 5. Geometric transforms

```python
from skimage.transform import AffineTransform, warp

t = np.array([[sx, 0,  0],     # sx – scale along x (columns)
              [0,  sy, 0],     # sy – scale along y (rows)
              [0,  0,  1]])    # homogeneous coordinates (always 0, 0, 1)

transform = AffineTransform(matrix=t)
transformed_img = warp(img, transform.inverse, order=0)   # order: 0 – nearest neighbour, 1 – linear
```

`warp` takes the **inverse** transform: for each output pixel it looks up where it came from in the input image.

### Reading the matrix
```
              × x   × y   const
new x  →   [  a     b     c  ]
new y  →   [  d     e     f  ]
           [  0     0     1  ]
```
Row = which coordinate changes, column = what it depends on.

### Typical matrices
```python
angle = np.deg2rad(15)                     # degrees → radians (np.sin/np.cos need radians)

rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0,              0,             1]])   # .T → rotation in the opposite direction

shear_x = np.array([[1, k, 0],             # x' = x + k·y
                    [0, 1, 0],
                    [0, 0, 1]])

translation = np.array([[1, 0, tx],        # x' = x + tx
                        [0, 1, ty],        # y' = y + ty
                        [0, 0, 1]])
```

### Manual transform of points (matrix multiplication)
```python
coords_h = np.column_stack((coords, np.ones(coords.shape[0])))   # (N, 2) → (N, 3): [x, y, 1]
new_coords = (A @ coords_h.T).T                                   # @ – matrix multiplication; result (N, 3)
x_new, y_new = new_coords[:, 0], new_coords[:, 1]
```

---

## 6. Interpolation (resampling a 1D signal)

```python
signal = img[12]                                         # one row of the image as a 1D signal
original_space = np.linspace(0, 99, signal.shape[0])     # where the original samples are
new_space = np.arange(0, 100)                            # where we want the new samples
```

### Nearest neighbour
```python
d = np.abs(new_space - original_space[:, np.newaxis])   # distance matrix (n_orig × n_new), via broadcasting
min_d = np.argmin(d, axis=0)                            # index of the nearest original sample for each new one
new_signal = signal[min_d]                              # values of the nearest neighbours
```

### Linear
```python
new_signal = np.interp(new_space, original_space, signal)   # built-in linear interpolation
```

### Nearest neighbour in 2D
```python
from scipy.spatial.distance import cdist

d = cdist(new_points, source_points)          # (n_new, 2) and (n_src, 2) → distance matrix (n_new × n_src)
nearest = np.argmin(d, axis=1)                # nearest source point for each new point
new_values = source_values[nearest]           # copy its value
```

---

## 7. Images

```python
import skimage

cat = skimage.data.cat()                 # colour image (H, W, 3), uint8 (0–255)
moon = skimage.data.moon()               # grayscale image (H, W)

mono = np.mean(cat, axis=2)              # colour → grayscale (average of R, G, B)
small = mono[::8, ::8]                   # 8× smaller in both dimensions
red = cat[:, :, 0]                       # single channel: 0 – R, 1 – G, 2 – B
```

---

## 8. Look-Up Table (LUT)

LUT = vector of `L` values (`L = 2**8 = 256`), integers from `0` to `L-1`. `lut[v]` is the new value for a pixel with value `v`.

```python
L = 2 ** 8
identity = np.linspace(0, L - 1, L).astype(int)    # 0, 1, ..., 255
negation = np.linspace(L - 1, 0, L).astype(int)    # 255, ..., 0

gamma = identity ** 0.3                            # any function of the identity...
gamma = gamma - np.min(gamma)
gamma = gamma / np.max(gamma)                      # ...normalize to 0–1
gamma = np.rint(gamma * (L - 1)).astype(int)       # ...scale to 0–255 and quantize

new_img = lut[img]          # apply LUT to every pixel (works on all channels at once)
```

```python
x_lut = np.arange(L)
ax.scatter(x_lut, lut, c='black', s=1)   # plot LUT: input on x, output on y
```

---

## 9. Histograms

```python
values, counts = np.unique(img, return_counts=True)   # only values that occur in the image
counts = np.bincount(img.flatten(), minlength=L)      # counts for ALL values 0..L-1 (missing → 0)
prob = counts / np.sum(counts)                        # probabilities (sum = 1)

ax.bar(np.arange(L), prob, color='black', width=1)    # histogram
```

For a colour image do it separately for `img[:, :, 0]`, `[:, :, 1]`, `[:, :, 2]`.

### Cumulative distribution and histogram equalization
```python
cdf = np.cumsum(prob)                        # cumulative sum: from ~0 to 1
lut = np.rint(cdf * (L - 1)).astype(int)     # equalization LUT
equalized = lut[img]                         # higher contrast
```

### Entropy
```python
from scipy.stats import entropy

h = entropy(prob)        # takes probabilities, not the image! higher = more information
```
---

## 10. Spatial filtering (correlation, convolution)

```python
from scipy import ndimage

sobel_s1 = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])          # gradient operator (edges)

out = ndimage.correlate(img, kernel)      # correlation
out = ndimage.convolve(img, kernel)       # convolution = correlation with kernel rotated by 180°
flipped = np.flip(kernel)                 # flip in both axes (vertical + horizontal)
```

Correlation for one pixel: place the kernel's centre on the pixel, multiply each kernel element by the pixel under it and sum.

```
out[i, j] = Σ Σ kernel[u + r, v + r] · img[i + u, j + v],   u, v = -r … r,   r = kernel.shape[0] // 2
```

For Sobel S1 `np.flip(S1) == -S1`, so `convolve` gives exactly `-correlate` (inverted relief).

### Own implementation (any odd kernel size)
```python
def correlate_self(image, kernel):
    r = kernel.shape[0] // 2                          # radius: 1 for 3×3, 3 for 7×7
    out = np.zeros_like(image, dtype=float)           # borders = 0 (np.copy(image).astype(float) → borders = original)
    for i in range(r, image.shape[0] - r):
        for j in range(r, image.shape[1] - r):
            pixel = 0
            for u in range(-r, r + 1):
                for v in range(-r, r + 1):
                    pixel = pixel + kernel[u + r, v + r] * image[i + u, j + v]
            out[i, j] = pixel
    return out

def convolve_self(image, kernel):
    return correlate_self(image, np.flip(kernel))
```

scipy handles borders by reflecting the image (`mode='reflect'`), so results differ from the own function only on the border pixels.

### Blur (mean filter)
```python
blur_kernel = np.ones((7, 7)) / 49     # constant coefficients, sum = 1 → brightness stays the same
blurred = correlate_self(img, blur_kernel)
```

### Unsharp masking (sharpening)
```python
mask = img - blurred          # details (edges); values around 0, positive and negative
sharpened = img + mask        # original + details

ax.imshow(sharpened, cmap='gray', vmin=0, vmax=255)   # vmin/vmax for images, not for the mask
```