#!/usr/bin/env python3
"""
Real data tests for the layer decomposition evaluator.

This test suite uses actual data from the project's test datasets to validate
the evaluator's performance on real-world scenarios.
"""

import unittest
import os
import glob
from PIL import Image
from typing import List, Dict, Any

# Import the modules we're testing
from src.evaluator import (
    LayerDecompositionEvaluator,
    evaluate_layer_decomposition
)
from src.render import render_layers


class TestEvaluatorWithRealData(unittest.TestCase):
    """Test evaluator with real data from the project datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LayerDecompositionEvaluator()
        
        # Define paths to test data
        self.gt_layers_path = "output_test/gt_layers"
        self.inputs_path = "output_test/inputs"
        
        # Check if test data exists
        self.has_gt_data = os.path.exists(self.gt_layers_path)
        self.has_inputs = os.path.exists(self.inputs_path)
    
    def load_gt_layers(self, sample_id: str) -> List[Image.Image]:
        """Load ground truth layers for a given sample ID."""
        sample_path = os.path.join(self.gt_layers_path, sample_id)
        if not os.path.exists(sample_path):
            return []
        
        # Find all layer files (*.png) and sort them numerically
        layer_files = sorted(glob.glob(os.path.join(sample_path, "*.png")),
                           key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        layers = []
        
        for layer_file in layer_files:
            try:
                layer = Image.open(layer_file).convert("RGBA")
                layers.append(layer)
            except Exception as e:
                print(f"Warning: Could not load layer {layer_file}: {e}")
        
        return layers
    
    def load_input_image(self, sample_id: str) -> Image.Image:
        """Load input image for a given sample ID."""
        input_file = os.path.join(self.inputs_path, f"{sample_id}.png")
        if not os.path.exists(input_file):
            return None
        
        try:
            return Image.open(input_file).convert("RGBA")
        except Exception as e:
            print(f"Warning: Could not load input {input_file}: {e}")
            return None
    
    def get_available_samples(self) -> List[str]:
        """Get list of sample IDs that have both GT layers and input images."""
        if not (self.has_gt_data and self.has_inputs):
            return []
        
        # Get sample IDs from GT layers directory
        gt_samples = set(d for d in os.listdir(self.gt_layers_path) 
                        if os.path.isdir(os.path.join(self.gt_layers_path, d)))
        
        # Get sample IDs from inputs directory
        input_samples = set(os.path.splitext(f)[0] for f in os.listdir(self.inputs_path) 
                           if f.endswith('.png'))
        
        # Return intersection (samples that have both GT and input)
        return sorted(list(gt_samples & input_samples))
    
    def test_perfect_reconstruction_with_real_data(self):
        """Test perfect reconstruction using real ground truth data."""
        if not (self.has_gt_data and self.has_inputs):
            self.skipTest("Real test data not available")
        
        available_samples = self.get_available_samples()
        if not available_samples:
            self.skipTest("No samples with both GT layers and input images found")
        
        # Test with the first available sample
        sample_id = available_samples[0]
        gt_layers = self.load_gt_layers(sample_id)
        input_image = self.load_input_image(sample_id)
        
        if len(gt_layers) < 2:
            self.skipTest(f"Sample {sample_id} has insufficient layers ({len(gt_layers)})")
        
        if input_image is None:
            self.skipTest(f"Could not load input image for sample {sample_id}")
        
        # Test perfect reconstruction (GT as predictions)
        result = self.evaluator.evaluate_decomposition(gt_layers, gt_layers, input_image)
        
        # Perfect reconstruction should have high scores
        self.assertGreater(result['total_score'], 0.8, 
                          f"Perfect reconstruction score too low: {result['total_score']}")
        self.assertGreater(result['global_reconstruction_score'], 0.8,
                          f"Perfect global reconstruction score too low: {result['global_reconstruction_score']}")
        
        print(f"✓ Perfect reconstruction test passed for sample {sample_id}")
        print(f"  Total score: {result['total_score']:.3f}")
        print(f"  Global reconstruction: {result['global_reconstruction_score']:.3f}")
        print(f"  Number of layers: {len(gt_layers)}")
        print(f"  Input image size: {input_image.size}")
    
    def test_missing_layers_with_real_data(self):
        """Test evaluation with missing layers using real data."""
        if not (self.has_gt_data and self.has_inputs):
            self.skipTest("Real test data not available")
        
        available_samples = self.get_available_samples()
        if not available_samples:
            self.skipTest("No samples found")
        
        # Find a sample with at least 3 layers
        sample_id = None
        for sid in available_samples:
            gt_layers = self.load_gt_layers(sid)
            if len(gt_layers) >= 3:
                sample_id = sid
                break
        
        if sample_id is None:
            self.skipTest("No samples with at least 3 layers found")
        
        gt_layers = self.load_gt_layers(sample_id)
        input_image = self.load_input_image(sample_id)
        
        if input_image is None:
            self.skipTest(f"Could not load input image for sample {sample_id}")
        
        # Test with missing layers (only predict first half)
        pred_layers = gt_layers[:len(gt_layers)//2]
        result = self.evaluator.evaluate_decomposition(pred_layers, gt_layers, input_image)
        
        # Missing layers should result in lower scores than perfect reconstruction
        perfect_result = self.evaluator.evaluate_decomposition(gt_layers, gt_layers, input_image)
        self.assertLess(result['total_score'], perfect_result['total_score'],
                       f"Missing layers score should be lower than perfect")
        
        # Check that some GT layers have empty subsets
        empty_subsets = sum(1 for gt_idx, subset_result in result['subset_results'].items()
                           if len(subset_result['predicted_indices']) == 0)
        self.assertGreater(empty_subsets, 0, "Expected some GT layers to have no predictions")
        
        print(f"✓ Missing layers test passed for sample {sample_id}")
        print(f"  Total score: {result['total_score']:.3f}")
        print(f"  Perfect score: {perfect_result['total_score']:.3f}")
        print(f"  Predicted {len(pred_layers)}/{len(gt_layers)} layers")
        print(f"  Empty subsets: {empty_subsets}")
    
    def test_cross_sample_evaluation(self):
        """Test evaluation using layers from different samples (should perform poorly)."""
        if not (self.has_gt_data and self.has_inputs):
            self.skipTest("Real test data not available")
        
        available_samples = self.get_available_samples()
        if len(available_samples) < 2:
            self.skipTest("Need at least 2 samples for cross-sample test")
        
        # Load layers from two different samples
        sample_id_1 = available_samples[0]
        sample_id_2 = available_samples[1]
        
        gt_layers_1 = self.load_gt_layers(sample_id_1)
        gt_layers_2 = self.load_gt_layers(sample_id_2)
        input_image_1 = self.load_input_image(sample_id_1)
        
        if len(gt_layers_1) < 2 or len(gt_layers_2) < 2:
            self.skipTest("Both samples need at least 2 layers")
        
        if input_image_1 is None:
            self.skipTest(f"Could not load input image for sample {sample_id_1}")
        
        # Resize all layers from sample 2 to match sample 1 dimensions
        target_size = gt_layers_1[0].size
        resized_gt_layers_2 = []
        for layer in gt_layers_2:
            resized_layer = layer.resize(target_size)
            resized_gt_layers_2.append(resized_layer)
        
        # This should perform poorly since layers are from different samples
        result = self.evaluator.evaluate_decomposition(resized_gt_layers_2, gt_layers_1, input_image_1)
        
        # Compare with perfect reconstruction
        perfect_result = self.evaluator.evaluate_decomposition(gt_layers_1, gt_layers_1, input_image_1)
        
        # Cross-sample evaluation should have lower scores
        self.assertLess(result['total_score'], perfect_result['total_score'],
                       f"Cross-sample score should be lower than perfect reconstruction")
        
        print(f"✓ Cross-sample test passed")
        print(f"  Sample 1 ({sample_id_1}): {len(gt_layers_1)} layers, size {gt_layers_1[0].size}")
        print(f"  Sample 2 ({sample_id_2}): {len(gt_layers_2)} layers, resized to {target_size}")
        print(f"  Cross-evaluation score: {result['total_score']:.3f}")
        print(f"  Perfect score: {perfect_result['total_score']:.3f}")
    
    def test_rendered_vs_input_consistency(self):
        """Test that rendered GT layers are consistent with input images."""
        if not (self.has_gt_data and self.has_inputs):
            self.skipTest("Real test data not available")
        
        available_samples = self.get_available_samples()
        if not available_samples:
            self.skipTest("No samples found")
        
        # Test with the first available sample
        sample_id = available_samples[0]
        gt_layers = self.load_gt_layers(sample_id)
        input_image = self.load_input_image(sample_id)
        
        if len(gt_layers) < 2:
            self.skipTest(f"Sample {sample_id} has insufficient layers")
        
        if input_image is None:
            self.skipTest(f"Could not load input image for sample {sample_id}")
        
        # Render GT layers and compare with input image
        rendered_gt = render_layers(gt_layers)
        
        # Resize if needed
        if rendered_gt.size != input_image.size:
            rendered_gt = rendered_gt.resize(input_image.size)
        
        # Compute similarity between rendered GT and input
        from src.utils.metrics import L1Loss
        l1_metric = L1Loss()
        similarity_loss = l1_metric.compute(rendered_gt, input_image)
        
        # They should be very similar (allowing for some compression artifacts)
        self.assertLess(similarity_loss, 0.1,
                       f"Rendered GT layers don't match input image well: {similarity_loss}")
        
        print(f"✓ Rendered vs input consistency test passed for sample {sample_id}")
        print(f"  L1 loss between rendered GT and input: {similarity_loss:.4f}")
        print(f"  Rendered size: {rendered_gt.size}")
        print(f"  Input size: {input_image.size}")
    
    def test_evaluator_consistency_real_data(self):
        """Test that evaluator produces consistent results across multiple runs with real data."""
        if not (self.has_gt_data and self.has_inputs):
            self.skipTest("Real test data not available")
        
        available_samples = self.get_available_samples()
        if not available_samples:
            self.skipTest("No samples found")
        
        # Test with the first available sample
        sample_id = available_samples[0]
        gt_layers = self.load_gt_layers(sample_id)
        input_image = self.load_input_image(sample_id)
        
        if len(gt_layers) < 2:
            self.skipTest(f"Sample {sample_id} has insufficient layers")
        
        if input_image is None:
            self.skipTest(f"Could not load input image for sample {sample_id}")
        
        # Run evaluation multiple times
        results = []
        for i in range(3):
            result = self.evaluator.evaluate_decomposition(gt_layers, gt_layers, input_image)
            results.append(result['total_score'])
        
        # Results should be identical (deterministic)
        for i in range(1, len(results)):
            self.assertAlmostEqual(results[0], results[i], places=6,
                                 msg=f"Inconsistent results: {results}")
        
        print(f"✓ Consistency test passed for sample {sample_id}")
        print(f"  Consistent score across {len(results)} runs: {results[0]:.6f}")
    
    def test_convenience_function_with_real_data(self):
        """Test the convenience function with real data."""
        if not (self.has_gt_data and self.has_inputs):
            self.skipTest("Real test data not available")
        
        available_samples = self.get_available_samples()
        if not available_samples:
            self.skipTest("No samples found")
        
        # Test with the first available sample
        sample_id = available_samples[0]
        gt_layers = self.load_gt_layers(sample_id)
        input_image = self.load_input_image(sample_id)
        
        if len(gt_layers) < 2:
            self.skipTest(f"Sample {sample_id} has insufficient layers")
        
        if input_image is None:
            self.skipTest(f"Could not load input image for sample {sample_id}")
        
        # Test convenience function
        result = evaluate_layer_decomposition(gt_layers, gt_layers, input_image)
        
        # Should produce same results as direct evaluator usage
        direct_result = self.evaluator.evaluate_decomposition(gt_layers, gt_layers, input_image)
        
        self.assertAlmostEqual(result['total_score'], direct_result['total_score'], places=6)
        self.assertAlmostEqual(result['global_reconstruction_score'], 
                              direct_result['global_reconstruction_score'], places=6)
        
        print(f"✓ Convenience function test passed for sample {sample_id}")
        print(f"  Score matches direct evaluator: {result['total_score']:.6f}")
    
    def test_multiple_samples_statistics(self):
        """Test evaluator on multiple samples and gather statistics."""
        if not (self.has_gt_data and self.has_inputs):
            self.skipTest("Real test data not available")
        
        available_samples = self.get_available_samples()
        if len(available_samples) < 3:
            self.skipTest("Need at least 3 samples for statistics")
        
        # Test on first 5 samples (or all if fewer)
        test_samples = available_samples[:min(5, len(available_samples))]
        
        perfect_scores = []
        layer_counts = []
        
        for sample_id in test_samples:
            gt_layers = self.load_gt_layers(sample_id)
            input_image = self.load_input_image(sample_id)
            
            if len(gt_layers) < 2 or input_image is None:
                continue
            
            result = self.evaluator.evaluate_decomposition(gt_layers, gt_layers, input_image)
            perfect_scores.append(result['total_score'])
            layer_counts.append(len(gt_layers))
        
        if not perfect_scores:
            self.skipTest("No valid samples found for statistics")
        
        # All perfect reconstructions should have high scores
        min_score = min(perfect_scores)
        max_score = max(perfect_scores)
        avg_score = sum(perfect_scores) / len(perfect_scores)
        
        self.assertGreater(min_score, 0.7, f"Minimum perfect score too low: {min_score}")
        
        print(f"✓ Multiple samples statistics test passed")
        print(f"  Tested {len(perfect_scores)} samples")
        print(f"  Perfect reconstruction scores: min={min_score:.3f}, max={max_score:.3f}, avg={avg_score:.3f}")
        print(f"  Layer counts: min={min(layer_counts)}, max={max(layer_counts)}, avg={sum(layer_counts)/len(layer_counts):.1f}")


def print_dataset_info():
    """Print information about available datasets."""
    print("Real Dataset Information:")
    print("=" * 50)
    
    # Check GT layers
    gt_path = "output_test/gt_layers"
    inputs_path = "output_test/inputs"
    
    if os.path.exists(gt_path):
        sample_ids = [d for d in os.listdir(gt_path) 
                     if os.path.isdir(os.path.join(gt_path, d))]
        print(f"Ground Truth Layers: {len(sample_ids)} samples")
        
        # Show layer count statistics
        layer_counts = []
        for sample_id in sample_ids[:10]:  # Check first 10
            sample_path = os.path.join(gt_path, sample_id)
            layer_count = len(glob.glob(os.path.join(sample_path, "*.png")))
            layer_counts.append(layer_count)
            if len(layer_counts) <= 5:  # Show first 5
                print(f"  - {sample_id}: {layer_count} layers")
        
        if len(sample_ids) > 5:
            print(f"  ... and {len(sample_ids) - 5} more samples")
        
        if layer_counts:
            print(f"  Layer count range: {min(layer_counts)}-{max(layer_counts)} (avg: {sum(layer_counts)/len(layer_counts):.1f})")
    else:
        print("Ground Truth Layers: Not found")
    
    if os.path.exists(inputs_path):
        input_files = [f for f in os.listdir(inputs_path) if f.endswith('.png')]
        print(f"Input Images: {len(input_files)} files")
    else:
        print("Input Images: Not found")
    
    # Check overlap
    if os.path.exists(gt_path) and os.path.exists(inputs_path):
        gt_samples = set(d for d in os.listdir(gt_path) 
                        if os.path.isdir(os.path.join(gt_path, d)))
        input_samples = set(os.path.splitext(f)[0] for f in os.listdir(inputs_path) 
                           if f.endswith('.png'))
        overlap = gt_samples & input_samples
        print(f"Samples with both GT and input: {len(overlap)}")
    
    print("=" * 50)


if __name__ == '__main__':
    print_dataset_info()
    print()
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluatorWithRealData)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Real Data Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if hasattr(result, 'skipped'):
        print(f"Skipped: {len(result.skipped)}")
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)