# Dimensionality reduction

This project explores principal component analysis (PCA) and Laplacian
Eigenmaps on synthetic data and handwritten digits. The PCA implementation
computes the mean, covariance, and projection using NumPy.

## Files

- `DR.py` loads the data, runs dimensionality reduction, and displays results.
- `pca_image_demo.py` interactively reconstructs handwritten digits with PCA.
- `data/` contains the Swiss roll and MNIST test datasets. The MNIST loader
  selects images of the digit 3.
- `results/` contains saved PCA means and projection matrices; `results/plots/`
  contains their visualizations.
- `feature_normalization.py` and `numpy_example.py` are standalone instructional
  examples. `collaboration.txt` records the collaboration statement.

## How to run

Use Python with NumPy and Matplotlib installed, and run commands from the
repository root. For example:

```sh
python DR.py --auto_check
python DR.py --data Swiss_Roll --method PCA --out_dim 2 --display --save
python DR.py --data toy_data --method LE --out_dim 2 --display
python DR.py --data MNIST --method PCA --out_dim 2 --display
```

`--path` changes the input data directory (default: `data`). For PCA,
`--display --save` writes the parameters to `results/` and any figure to
`results/plots/`, relative to the working directory. `--save` on its own does
not save anything: saving happens during display. Laplacian Eigenmaps does not
save results through this option, and MNIST has no LE visualization.

The built-in `--auto_check` checks the PCA mean and principal direction on a
small dataset; it is not a comprehensive test suite.

## Interactive image reconstruction

```sh
python pca_image_demo.py
```

The app fits PCA to the digit 3 images in the bundled MNIST test dataset, then
opens a window with sliders for the image and the number of components (up to
100). The first run may take a moment while PCA is computed. It needs an
interactive Matplotlib backend to show the window.

Variance retained describes the fraction of total variation in the fitted set
captured by the selected components. Mean squared error (MSE) compares the
selected original image with its reconstruction on the 0–255 pixel scale;
lower is better. The display clips pixel brightness to that range, but MSE uses
the unclipped reconstruction. This demo describes reconstruction quality, not a
measured file compression ratio.
