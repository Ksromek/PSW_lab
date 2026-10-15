## 1. Declaring arrays

```python
array = np.array([1, 2, 3])                    # array from a list
zero_array = np.zeros((n, m))                  # n × m array of zeros (shape as a tuple!)
zero_array_with_type = np.zeros((n, m), dtype=int)  # same, with integer type
array_copy = np.copy(array)                    # independent copy (changes don't affect the original)
```

### Ranges of values
```python
x = np.linspace(m, n, k)    # m – start, n – stop (included), k – number of samples
x = np.arange(m, n, s)      # m – start, n – stop (excluded), s – step (default 1)
```

### Random arrays
```python
img = np.random.choice(256, size=(m, n))   # random integers from 0 to 255, shape m × n
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
```

---

## 3. Figures and axes

```python
fig, axs = plt.subplots(m, n, figsize=(x, y))  # m – rows, n – columns; figsize in inches (width, height)

axs[0, 0].imshow(x, cmap='magma')   # x – what we want to show, cmap – colour map
axs[0, 0].set_title("Title")

fig.tight_layout()                  # fix spacing between subplots
fig.savefig("figure.png")           # save to file
```

### Other plot types and options
```python
plt.imshow(img, cmap='binary_r')                 # grayscale: 0 – black, max – white
plt.scatter(x, y, color='tomato')                # points instead of a line
plt.scatter(x, y, marker='x', color='b')         # custom marker and colour
plt.grid(ls=':')                                 # dotted grid
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
```

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