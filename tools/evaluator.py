"""
Layer decomposition evaluation framework.

This module provides comprehensive evaluation metrics for layer decomposition algorithms,
including reconstruction quality, layer redundancy, and fragmentation penalties.
"""

from typing import Any, Callable, Dict, List

import numpy as np
from PIL import Image
from tools.match import LayerMatcher
from tools.render import render_layers
from tools.utils.metrics import L1Loss, UnionMaskedL1Loss


def extract_alpha_mask(layer: Image.Image) -> np.ndarray:
    """
    Extract binary alpha mask from a PIL image.

    Args:
        layer: PIL Image (will be converted to RGBA if needed)

    Returns:
        Boolean numpy array of shape (H, W) with True where alpha > 0

    Example:
        >>> layer = Image.open("layer.png")
        >>> mask = extract_alpha_mask(layer)
        >>> visible_pixels = mask.sum()
    """
    if layer.mode != "RGBA":
        layer = layer.convert("RGBA")

    alpha_channel = np.asarray(layer.getchannel("A"), dtype=np.uint8)
    return alpha_channel > 0


class LayerRedundancyMetric:
    """
    Measures redundancy within a subset of layers.

    For each layer i in the subset, computes:
        redundancy_i = intersection(layer_i, union(other_layers)) / area(layer_i)

    The final redundancy is the mean across all layers in the subset.

    Interpretation:
        - 0.0: No redundancy (layers don't overlap)
        - 1.0: Complete redundancy (all layers fully covered by others)
        - Higher values indicate more "franken-layers" or unnecessary fragmentation
    """

    def __init__(self):
        """Initialize the redundancy metric."""
        self.metric_name = "layer_redundancy"

    def compute_redundancy(self, layers: List[Image.Image]) -> float:
        """
        Compute mean redundancy score for a list of layers.

        Args:
            layers: List of PIL Images to analyze for redundancy

        Returns:
            Mean redundancy score in range [0, 1]. Returns 0.0 if <= 1 layer
            or if all layers are empty.

        Example:
            >>> metric = LayerRedundancyMetric()
            >>> redundancy = metric.compute_redundancy([layer1, layer2, layer3])
        """
        if len(layers) <= 1:
            return 0.0

        # Extract alpha masks for all layers
        alpha_masks = [extract_alpha_mask(layer) for layer in layers]
        redundancy_scores: List[float] = []

        for current_index, current_mask in enumerate(alpha_masks):
            current_area = int(current_mask.sum())

            if current_area == 0:
                # Empty layer contributes 0 to redundancy
                redundancy_scores.append(0.0)
                continue

            # Get masks of all other layers
            other_masks = [
                mask for idx, mask in enumerate(alpha_masks) if idx != current_index
            ]

            # Compute union of all other layers
            union_of_others = np.logical_or.reduce(other_masks)

            # Compute intersection with union of others
            intersection_area = np.logical_and(current_mask, union_of_others).sum()
            layer_redundancy = intersection_area / current_area

            redundancy_scores.append(float(layer_redundancy))

        return float(np.mean(redundancy_scores)) if redundancy_scores else 0.0


class LayerDecompositionEvaluator:
    """
    Comprehensive evaluator for layer decomposition algorithms.

    This evaluator uses subset-based matching to compare predicted layers against
    ground truth layers. It computes multiple metrics including:

    1. Global reconstruction: How well all predicted layers reconstruct the input
    2. Subset reconstruction: How well each subset reconstructs its target GT layer
    3. Fragmentation penalty: Penalty for over-segmentation (multiple pred → 1 GT)
    4. Redundancy penalty: Penalty for overlapping layers within subsets

    The evaluation pipeline:
        1. Match predicted layers to GT layers using LayerMatcher
        2. For each GT layer, evaluate its assigned predicted subset
        3. Compute global reconstruction of all predicted layers
        4. Combine metrics into final scores
    """

    def __init__(
        self,
        layer_matcher: LayerMatcher = None,
        global_reconstruction_metric: Callable[
            [Image.Image, Image.Image], float
        ] = None,
        subset_reconstruction_metric: Callable[
            [Image.Image, Image.Image], float
        ] = None,
        redundancy_metric: LayerRedundancyMetric = None,
        subset_reconstruction_weight: float = 1.0,
        subset_redundancy_weight: float = 1.0,
        subset_fragmentation_weight: float = 1.0,
        global_reconstruction_weight: float = 0.1,
        layer_level_weight: float = 0.9,
    ):
        """
        Initialize the layer decomposition evaluator.

        Args:
            layer_matcher: LayerMatcher for computing layer similarities
            global_reconstruction_metric: Metric for global reconstruction loss
            subset_reconstruction_metric: Metric for subset reconstruction loss
            redundancy_metric: Metric for measuring layer redundancy
            subset_reconstruction_weight: Weight for subset reconstruction component
            subset_redundancy_weight: Weight for redundancy penalty component
            subset_fragmentation_weight: Weight for fragmentation penalty component
            global_reconstruction_weight: Weight for global reconstruction component
            layer_level_weight: Weight for layer-level losses (mean subset scores)
        """
        self.layer_matcher = layer_matcher or LayerMatcher()
        self.global_reconstruction_metric = global_reconstruction_metric or L1Loss()
        self.subset_reconstruction_metric = (
            subset_reconstruction_metric or UnionMaskedL1Loss()
        )
        self.redundancy_metric = redundancy_metric or LayerRedundancyMetric()

        # Component weights for final score computation
        self.subset_reconstruction_weight = subset_reconstruction_weight
        self.subset_redundancy_weight = subset_redundancy_weight
        self.subset_fragmentation_weight = subset_fragmentation_weight
        self.global_reconstruction_weight = global_reconstruction_weight
        self.layer_level_weight = layer_level_weight

    def _compute_subset_reconstruction_loss(
        self, predicted_subset: List[Image.Image], ground_truth_layer: Image.Image
    ) -> float:
        """
        Compute reconstruction loss for a predicted subset against a GT layer.

        Args:
            predicted_subset: List of predicted layers in the subset
            ground_truth_layer: Ground truth layer to compare against

        Returns:
            Reconstruction loss (lower is better). Returns infinity for empty subsets.
        """
        if not predicted_subset:
            return float("inf")

        rendered_subset = render_layers(predicted_subset)
        return self.subset_reconstruction_metric.compute(
            rendered_subset, ground_truth_layer
        )

    def _compute_fragmentation_penalty(self, subset_size: int) -> float:
        """
        Compute penalty for subset fragmentation (size > 1).

        Perfect matching has 1 predicted layer per GT layer. Larger subsets
        indicate over-segmentation and receive higher penalties.

        Formula: 1 - 1/k where k is subset size
        - k=1 → penalty=0.0 (perfect)
        - k=2 → penalty=0.5
        - k=3 → penalty=0.67
        - k→∞ → penalty→1.0

        Args:
            subset_size: Number of predicted layers in the subset

        Returns:
            Fragmentation penalty in range [0, 1)
        """
        if subset_size <= 1:
            return 0.0

        return float(1.0 - 1.0 / subset_size)

    def _compute_redundancy_penalty(self, predicted_subset: List[Image.Image]) -> float:
        """
        Compute redundancy penalty for layers within a subset.

        Args:
            predicted_subset: List of predicted layers in the subset

        Returns:
            Redundancy penalty in range [0, 1]
        """
        if len(predicted_subset) <= 1:
            return 0.0

        return self.redundancy_metric.compute_redundancy(predicted_subset)

    def _evaluate_single_subset(
        self, predicted_subset: List[Image.Image], ground_truth_layer: Image.Image
    ) -> Dict[str, float]:
        """
        Evaluate a single predicted subset against its matched GT layer.

        Args:
            predicted_subset: List of predicted layers in the subset
            ground_truth_layer: Ground truth layer to compare against

        Returns:
            Dictionary containing:
                - reconstruction_loss: Subset reconstruction quality
                - fragmentation_penalty: Penalty for over-segmentation
                - redundancy_penalty: Penalty for overlapping layers
                - total_score: Combined weighted score
                - subset_size: Number of layers in subset
        """
        # Handle empty subset case with worst possible scores
        if not predicted_subset:
            return {
                "reconstruction_loss": 1.0,
                "fragmentation_penalty": 1.0,
                "redundancy_penalty": 1.0,
                "total_score": 0.0,
                "subset_size": 0,
            }

        # Compute individual components
        reconstruction_loss = self._compute_subset_reconstruction_loss(
            predicted_subset, ground_truth_layer
        )
        fragmentation_penalty = self._compute_fragmentation_penalty(
            len(predicted_subset)
        )
        redundancy_penalty = self._compute_redundancy_penalty(predicted_subset)

        # Compute weighted total score: average of (1 - penalty) terms
        # Higher scores are better, so we use (1 - loss/penalty) for each component
        total_score = (
            self.subset_reconstruction_weight * (1.0 - reconstruction_loss)
            + self.subset_redundancy_weight * (1.0 - redundancy_penalty)
            + self.subset_fragmentation_weight * (1.0 - fragmentation_penalty)
        ) / 3.0

        return {
            "reconstruction_loss": reconstruction_loss,
            "fragmentation_penalty": fragmentation_penalty,
            "redundancy_penalty": redundancy_penalty,
            "total_score": total_score,
            "subset_size": len(predicted_subset),
        }

    def evaluate_decomposition(
        self,
        predicted_layers: List[Image.Image],
        ground_truth_layers: List[Image.Image],
        input_image: Image.Image,
    ) -> Dict[str, Any]:
        """
        Evaluate predicted layer decomposition against ground truth.

        Args:
            predicted_layers: List of predicted layer images
            ground_truth_layers: List of ground truth layer images
            input_image: Original input image for global reconstruction comparison

        Returns:
            Comprehensive evaluation results containing:
                - subset_results: Per-GT-layer evaluation results
                - total_score: Overall decomposition quality score
                - global_reconstruction_score: Global reconstruction quality

        Example:
            >>> evaluator = LayerDecompositionEvaluator()
            >>> results = evaluator.evaluate_decomposition(pred, gt, input_img)
            >>> print(f"Total score: {results['total_score']:.3f}")
        """
        # Step 1: Match predicted layers to ground truth layers
        layer_subsets = self.layer_matcher.match_layers_to_subsets(
            predicted_layers, ground_truth_layers
        )

        # Step 2: Compute global reconstruction quality
        if not predicted_layers:
            # No predictions means worst possible reconstruction
            global_reconstruction_loss = 1.0
        else:
            rendered_prediction = render_layers(predicted_layers)

            # Resize rendered prediction to match input image dimensions if needed
            if rendered_prediction.size != input_image.size:
                rendered_prediction = rendered_prediction.resize(input_image.size)

            global_reconstruction_loss = self.global_reconstruction_metric.compute(
                rendered_prediction, input_image
            )

        # Step 3: Evaluate each subset against its corresponding GT layer
        subset_evaluation_results: Dict[int, Dict[str, Any]] = {}
        subset_scores: List[float] = []

        for gt_index, predicted_indices in layer_subsets.items():
            ground_truth_layer = ground_truth_layers[gt_index]
            predicted_subset = [predicted_layers[i] for i in predicted_indices]

            # Evaluate this subset (handles empty subsets gracefully)
            subset_metrics = self._evaluate_single_subset(
                predicted_subset, ground_truth_layer
            )

            subset_evaluation_results[gt_index] = {
                "predicted_indices": predicted_indices,
                "metrics": subset_metrics,
            }

            subset_scores.append(subset_metrics["total_score"])

        # Step 4: Compute final aggregated scores
        mean_subset_score = sum(subset_scores) / len(subset_scores)
        global_reconstruction_score = 1.0 - global_reconstruction_loss

        # Final score: average of global reconstruction and sum of subset scores
        total_score = (
            self.global_reconstruction_weight * global_reconstruction_score
            + self.layer_level_weight * mean_subset_score
        )

        return {
            "subset_results": subset_evaluation_results,
            "total_score": total_score,
            "global_reconstruction_score": global_reconstruction_score,
        }


def evaluate_layer_decomposition(
    predicted_layers: List[Image.Image],
    ground_truth_layers: List[Image.Image],
    input_image: Image.Image,
    **evaluator_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for evaluating layer decomposition with default settings.

    Args:
        predicted_layers: List of predicted layer images
        ground_truth_layers: List of ground truth layer images
        input_image: Original input image for reconstruction comparison
        **evaluator_kwargs: Additional arguments for LayerDecompositionEvaluator

    Returns:
        Evaluation results dictionary from LayerDecompositionEvaluator

    Example:
        >>> results = evaluate_layer_decomposition(pred_layers, gt_layers, input_img)
        >>> print(f"Decomposition quality: {results['total_score']:.3f}")
    """
    evaluator = LayerDecompositionEvaluator(**evaluator_kwargs)
    return evaluator.evaluate_decomposition(
        predicted_layers, ground_truth_layers, input_image
    )
