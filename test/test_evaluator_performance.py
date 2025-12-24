#!/usr/bin/env python3
"""
Performance and edge case tests for the layer decomposition evaluator.

This test suite covers:
1. Performance benchmarks
2. Edge cases and boundary conditions
3. Stress tests with large numbers of layers
4. Memory usage validation
5. Error handling and robustness
"""

import unittest
import time
import numpy as np
from PIL import Image
import gc
import sys
from typing import List

# Import the modules we're testing
from src.evaluator import (
    extract_alpha_mask,
    LayerRedundancyMetric,
    LayerDecompositionEvaluator,
    evaluate_layer_decomposition
)
from src.render import render_layers


class TestEvaluatorPerformance(unittest.TestCase):
    """Test evaluator performance with various scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LayerDecompositionEvaluator()
    
    def create_test_layers(self, num_layers: int, size: tuple = (100, 100)) -> List[Image.Image]:
        """Create test layers with different patterns."""
        layers = []
        width, height = size
        
        for i in range(num_layers):
            # Create layer with different pattern for each
            layer_array = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Create different shapes for each layer
            if i % 4 == 0:  # Rectangle
                x1, y1 = i * 10 % (width - 20), i * 10 % (height - 20)
                layer_array[y1:y1+20, x1:x1+20, :] = [255, i*20 % 255, 0, 255]
            elif i % 4 == 1:  # Circle-like
                center_x, center_y = width // 2, height // 2
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= (10 + i*2)**2
                layer_array[mask] = [0, 255, i*20 % 255, 255]
            elif i % 4 == 2:  # Diagonal stripe
                for j in range(height):
                    start_x = (j + i*5) % width
                    end_x = min(start_x + 10, width)
                    layer_array[j, start_x:end_x, :] = [i*20 % 255, 0, 255, 255]
            else:  # Random pattern
                np.random.seed(i)  # Deterministic randomness
                mask = np.random.random((height, width)) > 0.7
                layer_array[mask] = [255, 255, i*20 % 255, 255]
            
            layers.append(Image.fromarray(layer_array, 'RGBA'))
        
        return layers
    
    def test_performance_small_layers(self):
        """Test performance with small number of layers."""
        num_layers = 5
        layers = self.create_test_layers(num_layers, (50, 50))
        input_image = render_layers(layers)
        
        start_time = time.time()
        result = self.evaluator.evaluate_decomposition(layers, layers, input_image)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete quickly for small inputs
        self.assertLess(execution_time, 1.0, f"Small layer evaluation took too long: {execution_time:.3f}s")
        self.assertGreater(result['total_score'], 0.9, "Perfect reconstruction should have high score")
        
        print(f"✓ Small layers performance test passed")
        print(f"  {num_layers} layers, 50x50 pixels")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Score: {result['total_score']:.3f}")
    
    def test_performance_medium_layers(self):
        """Test performance with medium number of layers."""
        num_layers = 15
        layers = self.create_test_layers(num_layers, (200, 200))
        input_image = render_layers(layers)
        
        start_time = time.time()
        result = self.evaluator.evaluate_decomposition(layers, layers, input_image)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should still be reasonably fast
        self.assertLess(execution_time, 5.0, f"Medium layer evaluation took too long: {execution_time:.3f}s")
        self.assertGreater(result['total_score'], 0.8, "Perfect reconstruction should have high score")
        
        print(f"✓ Medium layers performance test passed")
        print(f"  {num_layers} layers, 200x200 pixels")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Score: {result['total_score']:.3f}")
    
    def test_performance_many_layers(self):
        """Test performance with many layers."""
        num_layers = 30
        layers = self.create_test_layers(num_layers, (100, 100))
        input_image = render_layers(layers)
        
        start_time = time.time()
        result = self.evaluator.evaluate_decomposition(layers, layers, input_image)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should handle many layers without excessive slowdown
        self.assertLess(execution_time, 15.0, f"Many layer evaluation took too long: {execution_time:.3f}s")
        self.assertGreater(result['total_score'], 0.7, "Perfect reconstruction should have reasonable score")
        
        print(f"✓ Many layers performance test passed")
        print(f"  {num_layers} layers, 100x100 pixels")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Score: {result['total_score']:.3f}")
    
    def test_performance_large_images(self):
        """Test performance with large image dimensions."""
        num_layers = 8
        layers = self.create_test_layers(num_layers, (500, 500))
        input_image = render_layers(layers)
        
        start_time = time.time()
        result = self.evaluator.evaluate_decomposition(layers, layers, input_image)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Large images should still be manageable
        self.assertLess(execution_time, 10.0, f"Large image evaluation took too long: {execution_time:.3f}s")
        self.assertGreater(result['total_score'], 0.8, "Perfect reconstruction should have high score")
        
        print(f"✓ Large images performance test passed")
        print(f"  {num_layers} layers, 500x500 pixels")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Score: {result['total_score']:.3f}")
    
    def test_memory_usage_stability(self):
        """Test that memory usage doesn't grow excessively."""
        initial_objects = len(gc.get_objects())
        
        # Run multiple evaluations
        for i in range(5):
            layers = self.create_test_layers(10, (100, 100))
            input_image = render_layers(layers)
            result = self.evaluator.evaluate_decomposition(layers, layers, input_image)
            
            # Clean up explicitly
            del layers, input_image, result
            gc.collect()
        
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Memory growth should be reasonable
        self.assertLess(object_growth, 1000, f"Excessive object growth: {object_growth}")
        
        print(f"✓ Memory usage stability test passed")
        print(f"  Object growth after 5 evaluations: {object_growth}")


class TestEvaluatorEdgeCases(unittest.TestCase):
    """Test evaluator edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LayerDecompositionEvaluator()
    
    def test_single_pixel_layers(self):
        """Test with extremely small (1x1) layers."""
        # Create 1x1 layers
        layer1 = Image.new('RGBA', (1, 1), (255, 0, 0, 255))
        layer2 = Image.new('RGBA', (1, 1), (0, 255, 0, 255))
        input_image = Image.new('RGBA', (1, 1), (255, 0, 0, 255))
        
        result = self.evaluator.evaluate_decomposition([layer1], [layer1], input_image)
        
        # Should handle gracefully
        self.assertIn('total_score', result)
        self.assertGreaterEqual(result['total_score'], 0.0)
        self.assertLessEqual(result['total_score'], 1.0)
        
        print(f"✓ Single pixel layers test passed")
        print(f"  Score: {result['total_score']:.3f}")
    
    def test_very_large_layer_count(self):
        """Test with a very large number of layers."""
        num_layers = 50
        
        # Create layers that may overlap spatially but have different colors
        layers = []
        for i in range(num_layers):
            layer_array = np.zeros((100, 100, 4), dtype=np.uint8)
            
            # Create overlapping rectangles with different colors
            x = (i * 7) % 80  # Some overlap in x
            y = (i * 5) % 80  # Some overlap in y
            
            # Each layer has a unique color to avoid redundancy
            r = (i * 37) % 256
            g = (i * 73) % 256
            b = (i * 109) % 256
            
            layer_array[y:y+20, x:x+20, :] = [r, g, b, 255]
            layers.append(Image.fromarray(layer_array, 'RGBA'))
        
        input_image = render_layers(layers)
        
        # Test perfect reconstruction with many overlapping but differently colored layers
        result = self.evaluator.evaluate_decomposition(layers, layers, input_image)
        
        self.assertIn('total_score', result)
        self.assertGreater(result['total_score'], 0.95, f"Perfect reconstruction should have high score: {result['total_score']}")
        
        print(f"✓ Very large layer count test passed")
        print(f"  Tested with {len(layers)} overlapping layers with unique colors")
        print(f"  Score: {result['total_score']:.3f}")
    
    def test_all_transparent_layers(self):
        """Test with completely transparent layers."""
        # Create transparent layers
        transparent_layers = [
            Image.new('RGBA', (50, 50), (0, 0, 0, 0)),
            Image.new('RGBA', (50, 50), (255, 255, 255, 0)),
            Image.new('RGBA', (50, 50), (100, 100, 100, 0))
        ]
        
        input_image = Image.new('RGBA', (50, 50), (0, 0, 0, 0))
        
        result = self.evaluator.evaluate_decomposition(transparent_layers, transparent_layers, input_image)
        
        # Should handle gracefully
        self.assertIn('total_score', result)
        self.assertGreaterEqual(result['total_score'], 0.0)
        
        print(f"✓ All transparent layers test passed")
        print(f"  Score: {result['total_score']:.3f}")
    
    def test_identical_layers(self):
        """Test with multiple identical layers."""
        # Create identical layers
        base_layer = Image.new('RGBA', (50, 50), (255, 0, 0, 255))
        identical_layers = [base_layer.copy() for _ in range(5)]
        
        input_image = base_layer.copy()
        
        result = self.evaluator.evaluate_decomposition(identical_layers, [base_layer], input_image)
        
        # Should detect high redundancy
        self.assertIn('total_score', result)
        # Score should be lower due to redundancy
        self.assertLess(result['total_score'], 0.8)
        
        print(f"✓ Identical layers test passed")
        print(f"  Score with high redundancy: {result['total_score']:.3f}")
    
    def test_extreme_aspect_ratios(self):
        """Test with extreme aspect ratio images."""
        # Very wide image
        wide_layer = Image.new('RGBA', (500, 10), (255, 0, 0, 255))
        wide_input = wide_layer.copy()
        
        result = self.evaluator.evaluate_decomposition([wide_layer], [wide_layer], wide_input)
        self.assertIn('total_score', result)
        
        # Very tall image
        tall_layer = Image.new('RGBA', (10, 500), (0, 255, 0, 255))
        tall_input = tall_layer.copy()
        
        result = self.evaluator.evaluate_decomposition([tall_layer], [tall_layer], tall_input)
        self.assertIn('total_score', result)
        
        print(f"✓ Extreme aspect ratios test passed")
    
    def test_mixed_layer_sizes_with_resize(self):
        """Test evaluation when layers need resizing."""
        # Create layers of different sizes
        layer1 = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        
        # Input image with different size
        input_image = Image.new('RGBA', (150, 150), (255, 0, 0, 255))
        
        result = self.evaluator.evaluate_decomposition([layer1], [layer1], input_image)
        
        # Should handle resizing gracefully
        self.assertIn('total_score', result)
        self.assertGreaterEqual(result['total_score'], 0.0)
        
        print(f"✓ Mixed layer sizes test passed")
        print(f"  Score with resizing: {result['total_score']:.3f}")
    
    def test_numerical_stability(self):
        """Test numerical stability with edge case values."""
        # Create layers with very similar but not identical colors
        layer1_array = np.full((50, 50, 4), [100, 100, 100, 255], dtype=np.uint8)
        layer2_array = np.full((50, 50, 4), [101, 100, 100, 255], dtype=np.uint8)  # Very slight difference
        
        layer1 = Image.fromarray(layer1_array, 'RGBA')
        layer2 = Image.fromarray(layer2_array, 'RGBA')
        input_image = layer1.copy()
        
        result = self.evaluator.evaluate_decomposition([layer2], [layer1], input_image)
        
        # Should handle small differences gracefully
        self.assertIn('total_score', result)
        self.assertGreaterEqual(result['total_score'], 0.0)
        self.assertLessEqual(result['total_score'], 1.0)
        
        print(f"✓ Numerical stability test passed")
        print(f"  Score with tiny differences: {result['total_score']:.3f}")


class TestEvaluatorRobustness(unittest.TestCase):
    """Test evaluator robustness and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = LayerDecompositionEvaluator()
    
    def test_mismatched_prediction_gt_counts(self):
        """Test with different numbers of predicted vs GT layers."""
        # More predictions than GT
        pred_layers = [Image.new('RGBA', (50, 50), (255, 0, 0, 255)) for _ in range(5)]
        gt_layers = [Image.new('RGBA', (50, 50), (255, 0, 0, 255)) for _ in range(2)]
        input_image = Image.new('RGBA', (50, 50), (255, 0, 0, 255))
        
        result = self.evaluator.evaluate_decomposition(pred_layers, gt_layers, input_image)
        self.assertIn('total_score', result)
        
        # Fewer predictions than GT
        result = self.evaluator.evaluate_decomposition(gt_layers, pred_layers, input_image)
        self.assertIn('total_score', result)
        
        print(f"✓ Mismatched layer counts test passed")
    
    def test_different_color_modes(self):
        """Test with different image color modes."""
        # RGB layer (no alpha)
        rgb_layer = Image.new('RGB', (50, 50), (255, 0, 0))
        rgba_layer = Image.new('RGBA', (50, 50), (255, 0, 0, 255))
        input_image = rgba_layer.copy()
        
        # Should convert RGB to RGBA automatically
        result = self.evaluator.evaluate_decomposition([rgb_layer], [rgba_layer], input_image)
        self.assertIn('total_score', result)
        
        print(f"✓ Different color modes test passed")
        print(f"  RGB to RGBA conversion score: {result['total_score']:.3f}")
    
    def test_evaluator_consistency_across_runs(self):
        """Test that evaluator gives consistent results across multiple runs."""
        layers = [Image.new('RGBA', (50, 50), (255, 0, 0, 255))]
        input_image = layers[0].copy()
        
        scores = []
        for _ in range(10):
            result = self.evaluator.evaluate_decomposition(layers, layers, input_image)
            scores.append(result['total_score'])
        
        # All scores should be identical (deterministic)
        for score in scores[1:]:
            self.assertAlmostEqual(scores[0], score, places=10)
        
        print(f"✓ Consistency across runs test passed")
        print(f"  Consistent score: {scores[0]:.6f}")
    
    def test_custom_weights(self):
        """Test evaluator with custom component weights."""
        layers = [Image.new('RGBA', (50, 50), (255, 0, 0, 255))]
        input_image = layers[0].copy()
        
        # Test with different weight configurations
        weight_configs = [
            {'global_reconstruction_weight': 0.8, 'layer_level_weight': 0.2},
            {'subset_reconstruction_weight': 2.0, 'subset_redundancy_weight': 0.5},
            {'subset_fragmentation_weight': 0.1}
        ]
        
        for weights in weight_configs:
            evaluator = LayerDecompositionEvaluator(**weights)
            result = evaluator.evaluate_decomposition(layers, layers, input_image)
            self.assertIn('total_score', result)
            self.assertGreaterEqual(result['total_score'], 0.0)
        
        print(f"✓ Custom weights test passed")


def run_performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    evaluator = LayerDecompositionEvaluator()
    
    # Test different scenarios
    scenarios = [
        (5, (50, 50), "Small: 5 layers, 50x50"),
        (10, (100, 100), "Medium: 10 layers, 100x100"),
        (20, (100, 100), "Many layers: 20 layers, 100x100"),
        (5, (300, 300), "Large images: 5 layers, 300x300"),
    ]
    
    for num_layers, size, description in scenarios:
        # Create test data
        layers = []
        width, height = size
        for i in range(num_layers):
            layer_array = np.zeros((height, width, 4), dtype=np.uint8)
            x1, y1 = (i * 20) % (width - 30), (i * 15) % (height - 30)
            layer_array[y1:y1+20, x1:x1+20, :] = [255, i*50 % 255, 0, 255]
            layers.append(Image.fromarray(layer_array, 'RGBA'))
        
        input_image = render_layers(layers)
        
        # Benchmark
        start_time = time.time()
        result = evaluator.evaluate_decomposition(layers, layers, input_image)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"{description}")
        print(f"  Time: {execution_time:.3f}s")
        print(f"  Score: {result['total_score']:.3f}")
        print(f"  Memory objects: {len(gc.get_objects())}")
        print()


if __name__ == '__main__':
    # Run performance benchmark first
    run_performance_benchmark()
    
    # Run the test suite
    print("="*60)
    print("RUNNING PERFORMANCE AND EDGE CASE TESTS")
    print("="*60)
    
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEvaluatorPerformance,
        TestEvaluatorEdgeCases,
        TestEvaluatorRobustness
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Performance & Edge Case Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)