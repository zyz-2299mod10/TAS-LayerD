#!/usr/bin/env python3
"""
Comprehensive test suite for the layer decomposition evaluator.

This test suite covers:
1. Unit tests for individual components
2. Integration tests with synthetic data
3. Edge case testing
4. Performance validation
"""

import unittest
import numpy as np
from PIL import Image

# Import the modules we're testing
from src.evaluator import (
    extract_alpha_mask,
    LayerRedundancyMetric,
    LayerDecompositionEvaluator,
    evaluate_layer_decomposition,
)
from src.render import render_layers


class TestExtractAlphaMask(unittest.TestCase):
    """Test the extract_alpha_mask function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test images with different alpha patterns
        self.test_images = {}

        # Fully opaque image
        self.test_images["opaque"] = Image.new("RGBA", (10, 10), (255, 0, 0, 255))

        # Fully transparent image
        self.test_images["transparent"] = Image.new("RGBA", (10, 10), (255, 0, 0, 0))

        # Mixed alpha image
        mixed_array = np.zeros((10, 10, 4), dtype=np.uint8)
        mixed_array[:5, :, :] = [255, 0, 0, 255]  # Top half opaque
        mixed_array[5:, :, :] = [0, 255, 0, 0]  # Bottom half transparent
        self.test_images["mixed"] = Image.fromarray(mixed_array, "RGBA")

        # RGB image (no alpha)
        self.test_images["rgb"] = Image.new("RGB", (10, 10), (255, 0, 0))

    def test_extract_alpha_mask_opaque(self):
        """Test alpha mask extraction from fully opaque image."""
        mask = extract_alpha_mask(self.test_images["opaque"])

        self.assertEqual(mask.shape, (10, 10))
        self.assertTrue(np.all(mask))  # All pixels should be True
        self.assertEqual(mask.dtype, bool)

    def test_extract_alpha_mask_transparent(self):
        """Test alpha mask extraction from fully transparent image."""
        mask = extract_alpha_mask(self.test_images["transparent"])

        self.assertEqual(mask.shape, (10, 10))
        self.assertFalse(np.any(mask))  # All pixels should be False
        self.assertEqual(mask.dtype, bool)

    def test_extract_alpha_mask_mixed(self):
        """Test alpha mask extraction from mixed alpha image."""
        mask = extract_alpha_mask(self.test_images["mixed"])

        self.assertEqual(mask.shape, (10, 10))
        # Top half should be True, bottom half False
        self.assertTrue(np.all(mask[:5, :]))
        self.assertFalse(np.any(mask[5:, :]))

    def test_extract_alpha_mask_rgb_conversion(self):
        """Test alpha mask extraction from RGB image (should convert to RGBA)."""
        mask = extract_alpha_mask(self.test_images["rgb"])

        self.assertEqual(mask.shape, (10, 10))
        self.assertTrue(np.all(mask))  # RGB images become fully opaque when converted


class TestLayerRedundancyMetric(unittest.TestCase):
    """Test the LayerRedundancyMetric class."""

    def setUp(self):
        """Set up test fixtures."""
        self.metric = LayerRedundancyMetric()

        # Create test layers with known redundancy patterns
        self.layers = {}

        # Non-overlapping layers
        layer1_array = np.zeros((20, 20, 4), dtype=np.uint8)
        layer1_array[:10, :10, :] = [255, 0, 0, 255]  # Top-left red square
        self.layers["non_overlap_1"] = Image.fromarray(layer1_array, "RGBA")

        layer2_array = np.zeros((20, 20, 4), dtype=np.uint8)
        layer2_array[10:, 10:, :] = [0, 255, 0, 255]  # Bottom-right green square
        self.layers["non_overlap_2"] = Image.fromarray(layer2_array, "RGBA")

        # Fully overlapping layers
        layer3_array = np.zeros((20, 20, 4), dtype=np.uint8)
        layer3_array[5:15, 5:15, :] = [255, 0, 0, 255]  # Center red square
        self.layers["overlap_1"] = Image.fromarray(layer3_array, "RGBA")

        layer4_array = np.zeros((20, 20, 4), dtype=np.uint8)
        layer4_array[5:15, 5:15, :] = [0, 0, 255, 255]  # Same center blue square
        self.layers["overlap_2"] = Image.fromarray(layer4_array, "RGBA")

        # Partially overlapping layers
        layer5_array = np.zeros((20, 20, 4), dtype=np.uint8)
        layer5_array[0:15, 0:15, :] = [255, 255, 0, 255]  # Large yellow square
        self.layers["partial_1"] = Image.fromarray(layer5_array, "RGBA")

        layer6_array = np.zeros((20, 20, 4), dtype=np.uint8)
        layer6_array[5:20, 5:20, :] = [255, 0, 255, 255]  # Overlapping magenta square
        self.layers["partial_2"] = Image.fromarray(layer6_array, "RGBA")

        # Empty layer
        self.layers["empty"] = Image.new("RGBA", (20, 20), (0, 0, 0, 0))

    def test_redundancy_single_layer(self):
        """Test redundancy with single layer (should be 0)."""
        redundancy = self.metric.compute_redundancy([self.layers["non_overlap_1"]])
        self.assertEqual(redundancy, 0.0)

    def test_redundancy_empty_list(self):
        """Test redundancy with empty layer list (should be 0)."""
        redundancy = self.metric.compute_redundancy([])
        self.assertEqual(redundancy, 0.0)

    def test_redundancy_non_overlapping(self):
        """Test redundancy with non-overlapping layers (should be 0)."""
        layers = [self.layers["non_overlap_1"], self.layers["non_overlap_2"]]
        redundancy = self.metric.compute_redundancy(layers)
        self.assertEqual(redundancy, 0.0)

    def test_redundancy_fully_overlapping(self):
        """Test redundancy with fully overlapping layers (should be 1.0)."""
        layers = [self.layers["overlap_1"], self.layers["overlap_2"]]
        redundancy = self.metric.compute_redundancy(layers)
        self.assertEqual(redundancy, 1.0)

    def test_redundancy_partial_overlapping(self):
        """Test redundancy with partially overlapping layers."""
        layers = [self.layers["partial_1"], self.layers["partial_2"]]
        redundancy = self.metric.compute_redundancy(layers)

        # Should be between 0 and 1
        self.assertGreater(redundancy, 0.0)
        self.assertLess(redundancy, 1.0)

        # Calculate expected redundancy manually
        # partial_1: 15x15 = 225 pixels
        # partial_2: 15x15 = 225 pixels
        # overlap: 10x10 = 100 pixels
        # redundancy for each layer: 100/225 ≈ 0.444
        # mean redundancy: 0.444
        expected_redundancy = 100.0 / 225.0
        self.assertAlmostEqual(redundancy, expected_redundancy, places=3)

    def test_redundancy_with_empty_layer(self):
        """Test redundancy when one layer is empty."""
        layers = [self.layers["non_overlap_1"], self.layers["empty"]]
        redundancy = self.metric.compute_redundancy(layers)

        # Empty layer contributes 0, non-empty layer has no overlap with empty
        # Expected: (0 + 0) / 2 = 0
        self.assertEqual(redundancy, 0.0)


class TestLayerDecompositionEvaluatorMethods(unittest.TestCase):
    """Test individual methods of LayerDecompositionEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LayerDecompositionEvaluator()

        # Create simple test layers
        self.gt_layer = Image.new("RGBA", (50, 50), (255, 0, 0, 255))

        # Perfect prediction (identical to GT)
        self.perfect_pred = Image.new("RGBA", (50, 50), (255, 0, 0, 255))

        # Poor prediction (different color, same shape)
        self.poor_pred = Image.new("RGBA", (50, 50), (0, 255, 0, 255))

        # Empty prediction
        self.empty_pred = Image.new("RGBA", (50, 50), (0, 0, 0, 0))

    def test_compute_fragmentation_penalty(self):
        """Test fragmentation penalty calculation."""
        # Single layer (perfect) - no penalty
        penalty = self.evaluator._compute_fragmentation_penalty(1)
        self.assertEqual(penalty, 0.0)

        # Two layers - penalty = 1 - 1/2 = 0.5
        penalty = self.evaluator._compute_fragmentation_penalty(2)
        self.assertEqual(penalty, 0.5)

        # Three layers - penalty = 1 - 1/3 ≈ 0.667
        penalty = self.evaluator._compute_fragmentation_penalty(3)
        self.assertAlmostEqual(penalty, 2.0 / 3.0, places=3)

        # Zero layers - no penalty
        penalty = self.evaluator._compute_fragmentation_penalty(0)
        self.assertEqual(penalty, 0.0)

    def test_compute_redundancy_penalty(self):
        """Test redundancy penalty calculation."""
        # Single layer - no redundancy
        penalty = self.evaluator._compute_redundancy_penalty([self.perfect_pred])
        self.assertEqual(penalty, 0.0)

        # Empty list - no redundancy
        penalty = self.evaluator._compute_redundancy_penalty([])
        self.assertEqual(penalty, 0.0)

        # Two identical layers - full redundancy
        penalty = self.evaluator._compute_redundancy_penalty(
            [self.perfect_pred, self.perfect_pred]
        )
        self.assertEqual(penalty, 1.0)

    def test_compute_subset_reconstruction_loss_perfect(self):
        """Test subset reconstruction with perfect prediction."""
        loss = self.evaluator._compute_subset_reconstruction_loss(
            [self.perfect_pred], self.gt_layer
        )
        self.assertAlmostEqual(loss, 0.0, places=3)

    def test_compute_subset_reconstruction_loss_empty(self):
        """Test subset reconstruction with empty prediction."""
        loss = self.evaluator._compute_subset_reconstruction_loss([], self.gt_layer)
        self.assertEqual(loss, float("inf"))

    def test_evaluate_single_subset_perfect(self):
        """Test single subset evaluation with perfect prediction."""
        result = self.evaluator._evaluate_single_subset(
            [self.perfect_pred], self.gt_layer
        )

        self.assertIn("reconstruction_loss", result)
        self.assertIn("fragmentation_penalty", result)
        self.assertIn("redundancy_penalty", result)
        self.assertIn("total_score", result)
        self.assertIn("subset_size", result)

        # Perfect single prediction should have high score
        self.assertAlmostEqual(result["reconstruction_loss"], 0.0, places=2)
        self.assertEqual(result["fragmentation_penalty"], 0.0)
        self.assertEqual(result["redundancy_penalty"], 0.0)
        self.assertEqual(result["subset_size"], 1)
        self.assertGreater(result["total_score"], 0.9)

    def test_evaluate_single_subset_empty(self):
        """Test single subset evaluation with empty prediction."""
        result = self.evaluator._evaluate_single_subset([], self.gt_layer)

        # Empty subset should have worst possible scores
        self.assertEqual(result["reconstruction_loss"], 1.0)
        self.assertEqual(result["fragmentation_penalty"], 1.0)
        self.assertEqual(result["redundancy_penalty"], 1.0)
        self.assertEqual(result["total_score"], 0.0)
        self.assertEqual(result["subset_size"], 0)


class TestLayerDecompositionEvaluatorIntegration(unittest.TestCase):
    """Integration tests for the complete evaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LayerDecompositionEvaluator()

        # Create a simple 2-layer ground truth with larger coverage
        self.gt_layers = []

        # GT Layer 1: Red square covering left half
        gt1_array = np.zeros((100, 100, 4), dtype=np.uint8)
        gt1_array[10:90, 10:50, :] = [255, 0, 0, 255]
        self.gt_layers.append(Image.fromarray(gt1_array, "RGBA"))

        # GT Layer 2: Blue square covering right half
        gt2_array = np.zeros((100, 100, 4), dtype=np.uint8)
        gt2_array[10:90, 50:90, :] = [0, 0, 255, 255]
        self.gt_layers.append(Image.fromarray(gt2_array, "RGBA"))

        # Create input image by rendering GT layers
        self.input_image = render_layers(self.gt_layers)

    def test_perfect_decomposition(self):
        """Test evaluation with perfect layer decomposition."""
        # Use GT layers as predictions
        pred_layers = self.gt_layers.copy()

        result = self.evaluator.evaluate_decomposition(
            pred_layers, self.gt_layers, self.input_image
        )

        self.assertIn("subset_results", result)
        self.assertIn("total_score", result)
        self.assertIn("global_reconstruction_score", result)

        # Perfect decomposition should have high scores
        self.assertGreater(result["total_score"], 0.9)
        self.assertGreater(result["global_reconstruction_score"], 0.9)

        # Check subset results
        self.assertEqual(len(result["subset_results"]), 2)
        for gt_idx in range(2):
            self.assertIn(gt_idx, result["subset_results"])
            subset_result = result["subset_results"][gt_idx]
            self.assertIn("predicted_indices", subset_result)
            self.assertIn("metrics", subset_result)

    def test_over_segmented_decomposition(self):
        """Test evaluation with over-segmented layers."""
        # Split each GT layer into 2 parts
        pred_layers = []

        # Split GT layer 1 (left half [10:90, 10:50]) into top and bottom halves
        pred1a_array = np.zeros((100, 100, 4), dtype=np.uint8)
        pred1a_array[10:50, 10:50, :] = [255, 0, 0, 255]  # Top half of left
        pred_layers.append(Image.fromarray(pred1a_array, "RGBA"))

        pred1b_array = np.zeros((100, 100, 4), dtype=np.uint8)
        pred1b_array[50:90, 10:50, :] = [255, 0, 0, 255]  # Bottom half of left
        pred_layers.append(Image.fromarray(pred1b_array, "RGBA"))

        # Split GT layer 2 (right half [10:90, 50:90]) into top and bottom halves
        pred2a_array = np.zeros((100, 100, 4), dtype=np.uint8)
        pred2a_array[10:50, 50:90, :] = [0, 0, 255, 255]  # Top half of right
        pred_layers.append(Image.fromarray(pred2a_array, "RGBA"))

        pred2b_array = np.zeros((100, 100, 4), dtype=np.uint8)
        pred2b_array[50:90, 50:90, :] = [0, 0, 255, 255]  # Bottom half of right
        pred_layers.append(Image.fromarray(pred2b_array, "RGBA"))

        result = self.evaluator.evaluate_decomposition(
            pred_layers, self.gt_layers, self.input_image
        )

        # Should have good reconstruction but fragmentation penalty
        self.assertGreater(result["global_reconstruction_score"], 0.9)
        self.assertLess(result["total_score"], 0.9)  # Lower due to fragmentation

        # Each GT layer should be matched to 2 predicted layers
        for gt_idx in range(2):
            subset_result = result["subset_results"][gt_idx]
            self.assertEqual(len(subset_result["predicted_indices"]), 2)
            # Should have fragmentation penalty
            self.assertGreater(subset_result["metrics"]["fragmentation_penalty"], 0.0)

    def test_missing_layer_decomposition(self):
        """Test evaluation with missing layers."""
        # Only predict the first GT layer
        pred_layers = [self.gt_layers[0]]

        result = self.evaluator.evaluate_decomposition(
            pred_layers, self.gt_layers, self.input_image
        )

        # Should have moderate global reconstruction (missing half the image)
        # Note: Missing content isn't as bad as wrong content, so score won't be terrible
        self.assertLess(result["global_reconstruction_score"], 0.95)
        self.assertLess(result["total_score"], 0.8)

        # First GT layer should have a match, second should be empty
        self.assertEqual(len(result["subset_results"][0]["predicted_indices"]), 1)
        self.assertEqual(len(result["subset_results"][1]["predicted_indices"]), 0)

        # Empty subset should have worst scores
        empty_subset_metrics = result["subset_results"][1]["metrics"]
        self.assertEqual(empty_subset_metrics["total_score"], 0.0)


class TestConvenienceFunction(unittest.TestCase):
    """Test the convenience function."""

    def test_evaluate_layer_decomposition_function(self):
        """Test the convenience function works correctly."""
        # Create simple test data
        gt_layer = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        pred_layer = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        input_image = gt_layer.copy()

        result = evaluate_layer_decomposition([pred_layer], [gt_layer], input_image)

        self.assertIn("subset_results", result)
        self.assertIn("total_score", result)
        self.assertIn("global_reconstruction_score", result)
        self.assertGreater(result["total_score"], 0.9)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LayerDecompositionEvaluator()

    def test_empty_predictions(self):
        """Test evaluation with no predicted layers."""
        gt_layer = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        input_image = gt_layer.copy()

        result = self.evaluator.evaluate_decomposition([], [gt_layer], input_image)

        # Should handle gracefully with poor scores
        self.assertLess(result["total_score"], 0.1)
        self.assertEqual(len(result["subset_results"][0]["predicted_indices"]), 0)

    def test_extra_predictions(self):
        """Test evaluation with more predictions than ground truth layers."""
        gt_layer = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        pred_layer1 = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        pred_layer2 = Image.new("RGBA", (50, 50), (0, 255, 0, 255))  # Extra prediction
        input_image = gt_layer.copy()

        result = self.evaluator.evaluate_decomposition(
            [pred_layer1, pred_layer2], [gt_layer], input_image
        )

        # Should handle gracefully - both predictions will be assigned to the single GT layer
        self.assertIn("subset_results", result)
        self.assertIn("total_score", result)
        self.assertEqual(len(result["subset_results"][0]["predicted_indices"]), 2)

    def test_size_mismatch_handling(self):
        """Test handling of size mismatches between prediction and input."""
        gt_layer = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
        pred_layer = Image.new("RGBA", (50, 50), (255, 0, 0, 255))  # Same size as GT
        input_image = Image.new(
            "RGBA", (100, 100), (255, 0, 0, 255)
        )  # Different size input

        # Should handle input size mismatch by resizing rendered prediction
        result = self.evaluator.evaluate_decomposition(
            [pred_layer], [gt_layer], input_image
        )

        self.assertIn("total_score", result)
        # Score should be reasonable despite input size difference
        self.assertGreater(result["total_score"], 0.0)


def create_test_suite():
    """Create a comprehensive test suite."""
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestExtractAlphaMask,
        TestLayerRedundancyMetric,
        TestLayerDecompositionEvaluatorMethods,
        TestLayerDecompositionEvaluatorIntegration,
        TestConvenienceFunction,
        TestEdgeCases,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


if __name__ == "__main__":
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )
    print(f"{'=' * 50}")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
