"""
Layer rendering utilities for image composition.

This module provides functions for compositing multiple image layers into a single
rendered image, with proper alpha channel handling and transparency management.
"""

from typing import List

import numpy as np
from PIL import Image


def clean_rgba_layer(image: Image.Image) -> Image.Image:
    """
    Clean RGBA layer by zeroing RGB channels where alpha is transparent.

    This prevents color bleeding from transparent pixels during alpha compositing,
    which can cause unwanted darkening or color artifacts in the final render.

    Args:
        image: Input PIL Image to clean (will be converted to RGBA if needed)

    Returns:
        Cleaned RGBA image with RGB=[0,0,0] where alpha=0

    Example:
        >>> layer = Image.open("layer.png")
        >>> clean_layer = clean_rgba_layer(layer)
    """
    rgba_array = np.array(image.convert("RGBA"))
    alpha_channel = rgba_array[..., 3]

    # Set RGB to [0,0,0] where alpha is 0 to prevent color bleeding
    rgba_array[alpha_channel == 0] = [0, 0, 0, 0]

    return Image.fromarray(rgba_array, mode="RGBA")


def render_layers(layers: List[Image.Image]) -> Image.Image:
    """
    Composite multiple image layers into a single rendered image.

    Layers are composited from background to foreground using alpha blending.
    Each layer is cleaned before compositing to prevent color artifacts.

    Args:
        layers: List of PIL Images to composite, ordered from background to foreground.
                Must contain at least one layer.

    Returns:
        Final composited RGBA image

    Raises:
        ValueError: If no layers are provided for rendering

    Example:
        >>> background = Image.open("bg.png")
        >>> foreground = Image.open("fg.png")
        >>> result = render_layers([background, foreground])
        >>> result.save("composited.png")
    """
    if not layers:
        raise ValueError("Cannot render empty layer list - at least one layer required")

    # Initialize with cleaned background layer
    composited_image = clean_rgba_layer(layers[0])

    # Composite each subsequent layer on top using alpha blending
    for layer in layers[1:]:
        cleaned_layer = clean_rgba_layer(layer)
        composited_image = Image.alpha_composite(composited_image, cleaned_layer)

    return composited_image
