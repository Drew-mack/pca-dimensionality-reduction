"""Explore PCA reconstructions of handwritten 3s from the bundled MNIST data."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from DR import PCA, data_MNIST


def create_demo(X, max_components=100):
    """Create the figure and sliders for a matrix of flattened MNIST images."""
    max_components = min(max_components, X.shape[0], X.shape[1])
    print(f"Fitting PCA to {X.shape[1]} images; this may take a moment...")
    mu, components = PCA(X, max_components)
    centered = X - mu
    scores = components.T @ centered
    total_variance = np.sum(centered ** 2)
    if total_variance == 0:
        raise ValueError("PCA needs images with nonzero variation.")
    retained = np.cumsum(np.sum(scores ** 2, axis=1)) / total_variance

    fig, (original_ax, reconstructed_ax) = plt.subplots(1, 2, figsize=(9, 6))
    fig.suptitle("PCA image reconstruction: handwritten 3s", fontsize=16)
    fig.subplots_adjust(bottom=0.31, wspace=0.15, top=0.88)
    original_ax.set_title("Original")
    reconstructed_ax.set_title("PCA reconstruction")
    for ax in (original_ax, reconstructed_ax):
        ax.axis("off")
    original_image = original_ax.imshow(X[:, 0].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
    reconstructed_image = reconstructed_ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=255)
    metrics = fig.text(0.5, 0.28, "", ha="center", fontsize=11)
    fig.text(0.5, 0.22, "Variance retained is measured across all 3s used to fit PCA.",
             ha="center", fontsize=9)

    component_ax = fig.add_axes([0.23, 0.13, 0.62, 0.035])
    image_ax = fig.add_axes([0.23, 0.06, 0.62, 0.035])
    component_slider = Slider(component_ax, "Components", 1, max_components,
                              valinit=min(10, max_components), valstep=1, valfmt="%d")
    image_slider = Slider(image_ax, "Image", 1, X.shape[1], valinit=1, valstep=1, valfmt="%d")

    def update(_):
        count = int(component_slider.val)
        index = int(image_slider.val) - 1
        reconstruction = mu[:, 0] + components[:, :count] @ scores[:count, index]
        mse = np.mean((X[:, index] - reconstruction) ** 2)
        original_image.set_data(X[:, index].reshape(28, 28))
        reconstructed_image.set_data(reconstruction.reshape(28, 28))
        metrics.set_text(f"Variance retained: {retained[count - 1]:.1%}    |    "
                         f"Image reconstruction MSE: {mse:.1f}")
        fig.canvas.draw_idle()

    component_slider.on_changed(update)
    image_slider.on_changed(update)
    update(None)
    return fig, component_slider, image_slider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="Directory containing mnist_test.csv")
    args = parser.parse_args()
    X, _ = data_MNIST(SimpleNamespace(path=args.path))
    # Keep references to the sliders alive while the interactive window is open.
    fig, component_slider, image_slider = create_demo(X)
    plt.show()


if __name__ == "__main__":
    main()
