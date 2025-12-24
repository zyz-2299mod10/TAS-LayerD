"""
Crello dataset utilities for layer processing and manipulation.

This module provides functions for processing Crello dataset samples, including
layer preparation, image rotation, positioning, and alpha channel manipulation.
"""

from typing import List, Tuple

from PIL import Image, ImageEnhance


def prepare_layers_from_crello_sample(crello_sample: dict) -> List[Image.Image]:
    """
    Convert a Crello dataset sample into a list of positioned layer images.

    Takes raw Crello sample data and creates properly positioned, rotated, and
    scaled layer images on the canvas dimensions specified in the sample.

    Args:
        crello_sample: Dictionary containing Crello sample data with keys:
                      - 'image': List of layer images
                      - 'left', 'top': Layer positions
                      - 'width', 'height': Layer dimensions
                      - 'angle': Layer rotation angles
                      - 'canvas_width', 'canvas_height': Canvas dimensions

    Returns:
        List of PIL Images representing the positioned layers on the canvas

    Example:
        >>> sample = dataset['test'][0]  # Load from Crello dataset
        >>> layers = prepare_layers_from_crello_sample(sample)
        >>> print(f"Prepared {len(layers)} layers")
    """
    layer_images = crello_sample["image"]
    canvas_width = crello_sample["canvas_width"]
    canvas_height = crello_sample["canvas_height"]

    processed_layers = []

    # Process each layer with its positioning parameters
    layer_data = zip(
        layer_images,
        crello_sample["left"],
        crello_sample["top"],
        crello_sample["width"],
        crello_sample["height"],
        crello_sample["angle"],
    )

    for layer_image, left_pos, top_pos, width, height, rotation_angle in layer_data:
        # Resize layer to specified dimensions
        resized_layer = layer_image.resize((int(width), int(height)))

        # Create transparent canvas
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

        # Position and rotate layer on canvas
        positioned_canvas = paste_rotated_layer_on_canvas(
            canvas=canvas,
            layer_image=resized_layer,
            bounding_box=(int(left_pos), int(top_pos), int(width), int(height)),
            rotation_angle=rotation_angle,
            opacity=1.0,
        )

        # Convert alpha channel to binary and add to processed layers
        binary_alpha_layer = convert_alpha_to_binary(positioned_canvas.copy())
        processed_layers.append(binary_alpha_layer)

    return processed_layers


def paste_rotated_layer_on_canvas(
    canvas: Image.Image,
    layer_image: Image.Image,
    bounding_box: Tuple[int, int, int, int],
    rotation_angle: float,
    opacity: float,
) -> Image.Image:
    """
    Paste a rotated layer image onto a canvas at the specified position.

    The layer is positioned according to the bounding box, rotated by the specified
    angle, and composited with the given opacity.

    Args:
        canvas: Target PIL Image canvas to paste onto
        layer_image: Source PIL Image layer to be pasted
        bounding_box: Tuple (x, y, width, height) defining layer position and size
        rotation_angle: Rotation angle in degrees (counterclockwise)
        opacity: Layer opacity in range [0.0, 1.0]

    Returns:
        Canvas with the rotated layer pasted at the specified position

    Example:
        >>> canvas = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
        >>> layer = Image.open("layer.png")
        >>> result = paste_rotated_layer_on_canvas(canvas, layer, (100, 100, 200, 150), 45.0, 0.8)
    """
    x_position, y_position, layer_width, layer_height = bounding_box
    center_x, center_y = x_position + layer_width / 2, y_position + layer_height / 2

    # Resize layer to fit bounding box dimensions
    resized_layer = layer_image.resize((layer_width, layer_height))

    # Ensure RGBA format and apply binary alpha conversion
    rgba_layer = resized_layer.convert("RGBA")
    rgba_layer = convert_alpha_to_binary(rgba_layer)

    # Apply opacity adjustment to alpha channel
    red, green, blue, alpha = rgba_layer.split()
    adjusted_alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    rgba_layer.putalpha(adjusted_alpha)

    # Create transparent background for rotation
    transparent_background = Image.new(
        "RGBA", (layer_width, layer_height), (0, 0, 0, 0)
    )
    transparent_background.paste(rgba_layer, (0, 0), rgba_layer)

    # Rotate layer with expansion to prevent cropping
    rotated_layer = transparent_background.rotate(
        -1 * rotation_angle,  # Negative for counterclockwise rotation
        resample=Image.BILINEAR,
        expand=True,
    )

    # Calculate paste position to center the rotated layer
    rotated_width, rotated_height = rotated_layer.size
    paste_x = int(center_x - rotated_width / 2)
    paste_y = int(center_y - rotated_height / 2)

    # Paste rotated layer onto canvas
    canvas.paste(rotated_layer, (paste_x, paste_y), rotated_layer)

    return canvas


def convert_alpha_to_binary(image: Image.Image) -> Image.Image:
    """
    Convert alpha channel to binary (0 or 255) while preserving RGB channels.

    Applies a threshold of 127 to the alpha channel: pixels with alpha > 127
    become fully opaque (255), others become fully transparent (0).

    Args:
        image: PIL Image with alpha channel (will be converted to RGBA if needed)

    Returns:
        PIL Image with binary alpha channel and unchanged RGB channels

    Example:
        >>> layer = Image.open("layer_with_gradual_alpha.png")
        >>> binary_layer = convert_alpha_to_binary(layer)
        >>> # Alpha channel now contains only 0 and 255 values
    """
    # Ensure image is in RGBA format
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Split into individual channels
    red_channel, green_channel, blue_channel, alpha_channel = image.split()

    # Apply binary threshold to alpha channel
    ALPHA_THRESHOLD = 127
    binary_alpha = alpha_channel.point(
        lambda pixel_value: 255 if pixel_value > ALPHA_THRESHOLD else 0
    )

    # Recombine channels with binary alpha
    return Image.merge("RGBA", (red_channel, green_channel, blue_channel, binary_alpha))
