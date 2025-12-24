"""
Image comparison metrics for layer decomposition evaluation.

This module provides various loss functions and similarity metrics for comparing
predicted and ground truth images, with support for both RGB and RGBA formats.
"""

import numpy as np
from PIL import Image


class L1Loss:
    """
    L1 (Manhattan) distance loss for full-image comparison.

    Computes the mean absolute difference between RGB channels of two images.
    Commonly used for global reconstruction quality assessment.
    """

    def __init__(self):
        """Initialize L1 loss metric."""
        self.metric_name = "l1_loss"

    def compute(self, predicted_image: Image.Image, target_image: Image.Image) -> float:
        """
        Compute normalized L1 loss between two images.

        Both images are converted to RGB format before comparison to ensure
        consistent channel handling.

        Args:
            predicted_image: Predicted/reconstructed PIL Image
            target_image: Ground truth target PIL Image

        Returns:
            Normalized L1 loss in range [0, 1], where 0 = perfect match, 1 = maximum difference

        Example:
            >>> l1_metric = L1Loss()
            >>> loss = l1_metric.compute(predicted_img, ground_truth_img)
            >>> print(f"L1 loss: {loss:.4f}")
        """
        # Convert both images to RGB for consistent comparison
        predicted_rgb = predicted_image.convert("RGB")
        target_rgb = target_image.convert("RGB")

        # Convert to float arrays for computation
        predicted_array = np.asarray(predicted_rgb, dtype=np.float32)
        target_array = np.asarray(target_rgb, dtype=np.float32)

        # Compute mean absolute difference and normalize by max pixel value
        mean_absolute_difference = np.mean(np.abs(predicted_array - target_array))
        normalized_loss = mean_absolute_difference / 255.0

        return float(normalized_loss)


class UnionMaskedL1Loss:
    """
    Union-masked L1 loss for layer-specific comparison.

    This metric computes L1 loss only within the union of alpha masks from both images.
    It's designed to focus on regions where at least one image has visible content,
    making it ideal for comparing layer reconstructions where we care about both:
    - Missing content (present in target but not in prediction)
    - Spurious content (present in prediction but not in target)
    """

    def __init__(self, alpha_threshold: int = 0):
        """
        Initialize union-masked L1 loss.

        Args:
            alpha_threshold: Threshold for considering pixels as visible.
                           Pixels with alpha > threshold are considered visible.
        """
        self.alpha_threshold = alpha_threshold
        self.metric_name = "union_masked_l1_loss"

    def compute(self, predicted_image: Image.Image, target_image: Image.Image) -> float:
        """
        Compute union-masked L1 loss between two RGBA images.

        The loss is computed only within the union of alpha masks, focusing
        evaluation on regions where at least one image has visible content.

        Args:
            predicted_image: Predicted/reconstructed PIL Image (converted to RGBA)
            target_image: Ground truth target PIL Image (converted to RGBA)

        Returns:
            Normalized union-masked L1 loss in range [0, 1]. Returns 0.0 if
            both images are completely transparent.

        Example:
            >>> masked_l1 = UnionMaskedL1Loss(alpha_threshold=127)
            >>> loss = masked_l1.compute(pred_layer, gt_layer)
            >>> print(f"Masked L1 loss: {loss:.4f}")
        """
        # Convert both images to RGBA for alpha channel access
        predicted_rgba = predicted_image.convert("RGBA")
        target_rgba = target_image.convert("RGBA")

        # Convert to float arrays
        predicted_array = np.asarray(predicted_rgba, dtype=np.float32)
        target_array = np.asarray(target_rgba, dtype=np.float32)

        # Separate RGB and alpha channels
        predicted_rgb, predicted_alpha = (
            predicted_array[..., :3],
            predicted_array[..., 3],
        )
        target_rgb, target_alpha = target_array[..., :3], target_array[..., 3]

        # Create binary masks based on alpha threshold
        predicted_mask = predicted_alpha > self.alpha_threshold
        target_mask = target_alpha > self.alpha_threshold

        # Compute union of both masks
        union_mask = predicted_mask | target_mask

        # Handle case where both images are completely transparent
        if not np.any(union_mask):
            return 0.0

        # Compute RGB difference and normalize
        rgb_difference = np.abs(predicted_rgb - target_rgb) / 255.0

        # Broadcast union mask to match RGB dimensions (H, W, 3)
        union_mask_3d = np.broadcast_to(union_mask[..., None], rgb_difference.shape)

        # Extract differences only within union mask and compute mean
        masked_differences = rgb_difference[union_mask_3d].reshape(-1, 3)
        mean_masked_loss = masked_differences.mean()

        return float(mean_masked_loss)
