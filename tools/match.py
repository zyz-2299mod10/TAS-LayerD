"""
Layer matching utilities for comparing predicted and ground truth layers.

This module provides functionality to match predicted layers to ground truth layers
based on similarity metrics, enabling evaluation of layer decomposition algorithms.
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image


class LayerMatcher:
    """
    Matches predicted layers to ground truth layers based on similarity metrics.

    The matcher computes similarity between layers using configurable metrics
    and assigns each predicted layer to the most similar ground truth layer.
    """

    def __init__(
        self,
        similarity_function: Optional[
            Callable[[Image.Image, Image.Image], float]
        ] = None,
    ):
        """
        Initialize the layer matcher with a custom similarity function.

        Args:
            similarity_function: Custom function to compute layer similarity.
                                If None, uses default combined RGB L1 + alpha IoU metric.
                                Function should return higher values for more similar layers.
        """
        self.similarity_function = (
            similarity_function or self._compute_default_similarity
        )

    def _compute_rgb_l1_distance(
        self, rgb_array_1: np.ndarray, rgb_array_2: np.ndarray
    ) -> float:
        """
        Compute normalized L1 distance between RGB channels.

        Args:
            rgb_array_1: First RGB array of shape (H, W, 3)
            rgb_array_2: Second RGB array of shape (H, W, 3)

        Returns:
            Normalized L1 distance in range [0, 1], where 0 = identical, 1 = maximally different
        """
        rgb_l1_distance = np.mean(
            np.abs(rgb_array_1.astype(float) - rgb_array_2.astype(float))
        )
        return rgb_l1_distance / 255.0

    def _compute_alpha_intersection_over_union(
        self, alpha_array_1: np.ndarray, alpha_array_2: np.ndarray
    ) -> float:
        """
        Compute Intersection over Union (IoU) for alpha channel masks.

        Uses binary thresholding to convert alpha channels to masks, then computes
        the ratio of intersection to union areas.

        Args:
            alpha_array_1: First alpha channel array of shape (H, W)
            alpha_array_2: Second alpha channel array of shape (H, W)

        Returns:
            IoU score in range [0, 1], where 0 = no overlap, 1 = perfect overlap
        """
        ALPHA_THRESHOLD = 127

        # Create binary masks from alpha channels
        mask_1 = alpha_array_1 > ALPHA_THRESHOLD
        mask_2 = alpha_array_2 > ALPHA_THRESHOLD

        # Compute intersection and union
        intersection_area = np.logical_and(mask_1, mask_2).sum()
        union_area = np.logical_or(mask_1, mask_2).sum()

        # Handle edge case where both masks are empty
        return 0.0 if union_area == 0 else intersection_area / union_area

    def _compute_default_similarity(
        self, image_1: Image.Image, image_2: Image.Image
    ) -> float:
        """
        Compute default similarity metric combining RGB L1 distance and alpha IoU.

        The similarity combines:
        - RGB similarity: 1 - normalized_L1_distance
        - Alpha IoU: intersection_over_union of alpha masks

        Both components are weighted equally (0.5 each).

        Args:
            image_1: First PIL Image for comparison
            image_2: Second PIL Image for comparison

        Returns:
            Combined similarity score in range [0, 1], where 1 = most similar

        Raises:
            ValueError: If images have different shapes
        """
        # Convert both images to RGBA format for consistent processing
        rgba_image_1 = image_1.convert("RGBA")
        rgba_image_2 = image_2.convert("RGBA")

        # Convert to numpy arrays
        array_1, array_2 = np.array(rgba_image_1), np.array(rgba_image_2)

        # Validate image dimensions match
        if array_1.shape != array_2.shape:
            raise ValueError(
                f"Images must have identical dimensions for comparison. "
                f"Got shapes {array_1.shape} and {array_2.shape}"
            )

        # Extract RGB and alpha channels
        rgb_1, alpha_1 = array_1[..., :3], array_1[..., 3]
        rgb_2, alpha_2 = array_2[..., :3], array_2[..., 3]

        # Compute individual similarity components
        rgb_l1_distance = self._compute_rgb_l1_distance(rgb_1, rgb_2)
        alpha_iou = self._compute_alpha_intersection_over_union(alpha_1, alpha_2)

        # Convert RGB distance to similarity (higher = more similar)
        rgb_similarity = 1.0 - rgb_l1_distance

        # Combine RGB similarity and alpha IoU with equal weighting
        combined_similarity = 0.5 * rgb_similarity + 0.5 * alpha_iou

        return combined_similarity

    def match_layers_to_subsets(
        self,
        predicted_layers: List[Image.Image],
        ground_truth_layers: List[Image.Image],
    ) -> Dict[int, List[int]]:
        """
        Match predicted layers to ground truth layers, creating subsets.

        Each predicted layer is assigned to the ground truth layer with highest
        similarity. Multiple predicted layers can be assigned to the same ground
        truth layer (over-segmentation), and some ground truth layers may receive
        no predicted layers (missing layers).

        Args:
            predicted_layers: List of predicted layer images
            ground_truth_layers: List of ground truth layer images

        Returns:
            Dictionary mapping ground truth layer indices to lists of predicted
            layer indices. Keys are GT indices [0, len(gt_layers)-1], values are
            lists of predicted indices assigned to that GT layer (may be empty).

        Example:
            >>> matcher = LayerMatcher()
            >>> subsets = matcher.match_layers_to_subsets(pred_layers, gt_layers)
            >>> # subsets[0] = [1, 3] means predicted layers 1 and 3 match GT layer 0
        """
        # Initialize empty subsets for all ground truth layers
        layer_subsets = {gt_index: [] for gt_index in range(len(ground_truth_layers))}

        # Assign each predicted layer to its best matching ground truth layer
        for pred_index, predicted_layer in enumerate(predicted_layers):
            # Find ground truth layer with highest similarity to this predicted layer
            best_matching_gt_index = max(
                range(len(ground_truth_layers)),
                key=lambda gt_index: self.similarity_function(
                    predicted_layer, ground_truth_layers[gt_index]
                ),
            )

            # Add this predicted layer to the best matching GT layer's subset
            layer_subsets[best_matching_gt_index].append(pred_index)

        return layer_subsets
