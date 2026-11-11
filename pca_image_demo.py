"""Explore PCA reconstructions of handwritten 3s from the bundled MNIST data."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
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

    background = "#101827"
    foreground = "#edf3fa"
    muted = "#a8b7c9"
    accent = "#50d4bc"
    fig = plt.figure(figsize=(12, 9), facecolor=background)
    fig.text(0.06, 0.95, "PCA EXPLORER", color=accent, fontsize=10)
    fig.text(0.06, 0.91, "How much makes a 3?", color=foreground, fontsize=25)

    original_ax = fig.add_axes([0.14, 0.54, 0.29, 0.30])
    reconstructed_ax = fig.add_axes([0.57, 0.54, 0.29, 0.30])
    original_ax.set_title("Original", color=muted, fontsize=11, pad=10)
    reconstructed_ax.set_title("Reconstruction", color=muted, fontsize=11, pad=10)
    for ax in (original_ax, reconstructed_ax):
        ax.axis("off")
    original_image = original_ax.imshow(X[:, 0].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
    reconstructed_image = reconstructed_ax.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=255)
    metrics = fig.text(0.5, 0.50, "", ha="center", color=foreground, fontsize=11)
    fig.text(0.5, 0.475, "Variance is measured across all 3s used to fit PCA.",
             ha="center", color=muted, fontsize=9)

    fig.text(0.06, 0.43, "The building blocks", color=foreground, fontsize=14)
    fig.text(0.06, 0.405, "Mean image + weighted components = reconstruction",
             color=foreground, fontsize=11)
    gallery_size = min(5, max_components)
    gallery_left = 0.08
    gallery_width = 0.105
    gallery_gap = 0.045
    mean_ax = fig.add_axes([gallery_left, 0.255, gallery_width, 0.105])
    mean_ax.imshow(mu[:, 0].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
    mean_ax.set_title("Mean", color=muted, fontsize=10, pad=6)
    mean_ax.axis("off")
    weight_labels = []
    for i in range(gallery_size):
        ax = fig.add_axes([gallery_left + (i + 1) * (gallery_width + gallery_gap),
                           0.255, gallery_width, 0.105])
        component = components[:, i].reshape(28, 28)
        limit = np.max(np.abs(component)) or 1
        ax.imshow(component, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"PC {i + 1}", color=muted, fontsize=10, pad=6)
        ax.axis("off")
        weight_labels.append(fig.text(
            gallery_left + (i + 1) * (gallery_width + gallery_gap) + gallery_width / 2,
            0.245, "", ha="center", va="top", fontsize=9,
        ))
    fig.text(0.06, 0.195,
             "Weights are for the selected image. Red and blue show opposite pixel changes; "
             "each PC has its own color scale.",
             color=muted, fontsize=9)

    component_ax = fig.add_axes([0.16, 0.14, 0.68, 0.008], facecolor=background)
    image_ax = fig.add_axes([0.16, 0.065, 0.68, 0.008], facecolor=background)
    slider_style = {
        "color": accent,
        "track_color": "#344356",
        "initcolor": "none",
        "handle_style": {"facecolor": accent, "edgecolor": background, "size": 13},
    }
    component_slider = Slider(component_ax, "Components", 0, max_components,
                              valinit=min(10, max_components), valstep=1, valfmt="%d",
                              **slider_style)
    image_slider = Slider(image_ax, "Image", 1, X.shape[1], valinit=1, valstep=1, valfmt="%d",
                          **slider_style)
    for slider in (component_slider, image_slider):
        slider.label.set_color(foreground)
        slider.label.set_position((0, 3.3))
        slider.label.set_horizontalalignment("left")
        slider.label.set_fontsize(10)
        slider.valtext.set_color(accent)
        slider.valtext.set_position((1, 3.3))
        slider.valtext.set_horizontalalignment("right")
        slider.valtext.set_fontsize(10)

    # Prefer the system's Avenir Next, with a bundled Matplotlib fallback.
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "Avenir Next" if "Avenir Next" in available_fonts else "DejaVu Sans"
    for text in fig.findobj(match=Text):
        text.set_fontfamily(font_family)

    def update(_):
        count = int(component_slider.val)
        index = int(image_slider.val) - 1
        reconstruction = mu[:, 0] + components[:, :count] @ scores[:count, index]
        mse = np.mean((X[:, index] - reconstruction) ** 2)
        original_image.set_data(X[:, index].reshape(28, 28))
        reconstructed_image.set_data(reconstruction.reshape(28, 28))
        variance = retained[count - 1] if count else 0.0
        for i, label in enumerate(weight_labels):
            included = i < count
            label.set_text(f"{scores[i, index]:+.1f}" + ("" if included else "\nnot used"))
            label.set_color(accent if included else muted)
        metrics.set_text(f"Variance retained: {variance:.1%}    |    "
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
