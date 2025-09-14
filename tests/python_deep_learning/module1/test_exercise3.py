import sys
import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, Any

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner


class Exercise3Validator(TestValidator):
    """Validator for Module 1 Exercise 3: First Step with MLP"""
    
    # Section 1: Understanding nn.Linear
    def test_linear_layer_1(self, variables: Dict[str, Any]) -> bool:
        """Test if linear_layer_1 is correctly created"""
        if not self.check_variable(variables, 'linear_layer_1', nn.Linear):
            return False
        
        layer = variables['linear_layer_1']
        if layer.in_features != 10 or layer.out_features != 5:
            self.fail(f"Expected Linear(10, 5), got Linear({layer.in_features}, {layer.out_features})")
            return False
        
        return True
    
    def test_linear_layer_2(self, variables: Dict[str, Any]) -> bool:
        """Test if linear_layer_2 is correctly created"""
        if not self.check_variable(variables, 'linear_layer_2', nn.Linear):
            return False
        
        layer = variables['linear_layer_2']
        if layer.in_features != 5 or layer.out_features != 3:
            self.fail(f"Expected Linear(5, 3), got Linear({layer.in_features}, {layer.out_features})")
            return False
        
        return True
    
    def test_num_params_layer2(self, variables: Dict[str, Any]) -> bool:
        """Test if num_params_layer2 is correctly calculated"""
        if not self.check_variable(variables, 'num_params_layer2', int):
            return False
        
        # Parameters = (5 * 3) + 3 = 18
        expected = 18
        actual = variables['num_params_layer2']
        
        if actual != expected:
            self.fail(f"Expected {expected} parameters, got {actual}")
            return False
        
        return True
    
    # Section 2: Activation Functions
    def test_relu_activation(self, variables: Dict[str, Any]) -> bool:
        """Test if ReLU activation is created"""
        if not self.check_variable(variables, 'relu_activation', nn.ReLU):
            return False
        return True
    
    def test_sigmoid_activation(self, variables: Dict[str, Any]) -> bool:
        """Test if Sigmoid activation is created"""
        if not self.check_variable(variables, 'sigmoid_activation', nn.Sigmoid):
            return False
        return True
    
    def test_tanh_activation(self, variables: Dict[str, Any]) -> bool:
        """Test if Tanh activation is created"""
        if not self.check_variable(variables, 'tanh_activation', nn.Tanh):
            return False
        return True
    
    def test_linear_output(self, variables: Dict[str, Any]) -> bool:
        """Test if linear_output is correctly computed"""
        if not self.check_variable(variables, 'linear_output', torch.Tensor):
            return False
        
        output = variables['linear_output']
        expected_shape = (2, 5)  # Batch size 2, output features 5
        
        if tuple(output.shape) != expected_shape:
            self.fail(f"Expected shape {expected_shape}, got {tuple(output.shape)}")
            return False
        
        return True
    
    def test_activated_output(self, variables: Dict[str, Any]) -> bool:
        """Test if activated_output is correctly computed with ReLU"""
        if not self.check_variable(variables, 'activated_output', torch.Tensor):
            return False
        
        output = variables['activated_output']
        expected_shape = (2, 5)
        
        if tuple(output.shape) != expected_shape:
            self.fail(f"Expected shape {expected_shape}, got {tuple(output.shape)}")
            return False
        
        # Check that all values are non-negative (ReLU property)
        if (output < 0).any():
            self.fail("ReLU output should not contain negative values")
            return False
        
        return True
    
    # Section 3: Building Networks with nn.Sequential
    def test_simple_mlp(self, variables: Dict[str, Any]) -> bool:
        """Test if simple_mlp is correctly created"""
        if not self.check_variable(variables, 'simple_mlp', nn.Sequential):
            return False
        
        mlp = variables['simple_mlp']
        
        # Check number of layers
        if len(mlp) != 3:
            self.fail(f"Expected 3 layers (Linear, ReLU, Linear), got {len(mlp)}")
            return False
        
        # Check layer types and dimensions
        if not isinstance(mlp[0], nn.Linear) or mlp[0].in_features != 8 or mlp[0].out_features != 4:
            self.fail("First layer should be Linear(8, 4)")
            return False
        
        if not isinstance(mlp[1], nn.ReLU):
            self.fail("Second layer should be ReLU")
            return False
        
        if not isinstance(mlp[2], nn.Linear) or mlp[2].in_features != 4 or mlp[2].out_features != 2:
            self.fail("Third layer should be Linear(4, 2)")
            return False
        
        return True
    
    def test_deep_mlp(self, variables: Dict[str, Any]) -> bool:
        """Test if deep_mlp is correctly created"""
        if not self.check_variable(variables, 'deep_mlp', nn.Sequential):
            return False
        
        mlp = variables['deep_mlp']
        
        # Check number of layers (4 Linear + 3 ReLU = 7)
        if len(mlp) != 7:
            self.fail(f"Expected 7 layers, got {len(mlp)}")
            return False
        
        # Check first layer
        if not isinstance(mlp[0], nn.Linear) or mlp[0].in_features != 10 or mlp[0].out_features != 8:
            self.fail("First layer should be Linear(10, 8)")
            return False
        
        # Check last layer
        if not isinstance(mlp[6], nn.Linear) or mlp[6].in_features != 4 or mlp[6].out_features != 2:
            self.fail("Last layer should be Linear(4, 2)")
            return False
        
        return True
    
    def test_deep_mlp_params(self, variables: Dict[str, Any]) -> bool:
        """Test if deep_mlp_params is correctly calculated"""
        if not self.check_variable(variables, 'deep_mlp_params', int):
            return False
        
        # (10*8 + 8) + (8*6 + 6) + (6*4 + 4) + (4*2 + 2) = 88 + 54 + 28 + 10 = 180
        expected = 180
        actual = variables['deep_mlp_params']
        
        if actual != expected:
            self.fail(f"Expected {expected} parameters, got {actual}")
            return False
        
        return True
    
    # Section 4: Forward Pass
    def test_simple_output(self, variables: Dict[str, Any]) -> bool:
        """Test if simple_output is correctly computed"""
        if not self.check_variable(variables, 'simple_output', torch.Tensor):
            return False
        
        output = variables['simple_output']
        expected_shape = (3, 2)  # Batch size 3, output features 2
        
        if tuple(output.shape) != expected_shape:
            self.fail(f"Expected shape {expected_shape}, got {tuple(output.shape)}")
            return False
        
        return True
    
    def test_mixed_activation_mlp(self, variables: Dict[str, Any]) -> bool:
        """Test if mixed_activation_mlp is correctly created"""
        if not self.check_variable(variables, 'mixed_activation_mlp', nn.Sequential):
            return False
        
        mlp = variables['mixed_activation_mlp']
        
        # Check number of layers
        if len(mlp) != 6:
            self.fail(f"Expected 6 layers, got {len(mlp)}")
            return False
        
        # Check layer types
        expected_types = [nn.Linear, nn.ReLU, nn.Linear, nn.Tanh, nn.Linear, nn.Sigmoid]
        for i, expected_type in enumerate(expected_types):
            if not isinstance(mlp[i], expected_type):
                self.fail(f"Layer {i} should be {expected_type.__name__}")
                return False
        
        return True
    
    def test_mixed_output(self, variables: Dict[str, Any]) -> bool:
        """Test if mixed_output is correctly computed"""
        if not self.check_variable(variables, 'mixed_output', torch.Tensor):
            return False
        
        output = variables['mixed_output']
        expected_shape = (5, 1)  # Batch size 5, output features 1
        
        if tuple(output.shape) != expected_shape:
            self.fail(f"Expected shape {expected_shape}, got {tuple(output.shape)}")
            return False
        
        # Check that values are between 0 and 1 (Sigmoid output)
        if output.min() < 0 or output.max() > 1:
            self.fail("Sigmoid output should be between 0 and 1")
            return False
        
        return True
    
    # Section 5: Understanding Parameter Counting
    def test_count_parameters_function(self, variables: Dict[str, Any]) -> bool:
        """Test if count_parameters function works correctly"""
        if 'count_parameters' not in variables:
            self.fail("count_parameters function not found")
            return False
        
        count_fn = variables['count_parameters']
        
        # Test with a simple model
        test_model = nn.Linear(3, 2)
        expected = 3 * 2 + 2  # 8 parameters
        
        try:
            result = count_fn(test_model)
            if result != expected:
                self.fail(f"count_parameters returned {result}, expected {expected}")
                return False
        except Exception as e:
            self.fail(f"count_parameters raised exception: {e}")
            return False
        
        return True
    
    def test_large_mlp(self, variables: Dict[str, Any]) -> bool:
        """Test if large_mlp is correctly created"""
        if not self.check_variable(variables, 'large_mlp', nn.Sequential):
            return False
        
        mlp = variables['large_mlp']
        
        # Check that it has the right structure
        linear_layers = [layer for layer in mlp if isinstance(layer, nn.Linear)]
        
        if len(linear_layers) != 4:
            self.fail(f"Expected 4 Linear layers, got {len(linear_layers)}")
            return False
        
        # Check dimensions
        expected_dims = [(100, 64), (64, 32), (32, 16), (16, 10)]
        for i, (layer, (in_f, out_f)) in enumerate(zip(linear_layers, expected_dims)):
            if layer.in_features != in_f or layer.out_features != out_f:
                self.fail(f"Layer {i}: expected Linear({in_f}, {out_f}), got Linear({layer.in_features}, {layer.out_features})")
                return False
        
        return True
    
    def test_expected_params(self, variables: Dict[str, Any]) -> bool:
        """Test if expected_params is correctly calculated"""
        if not self.check_variable(variables, 'expected_params', int):
            return False
        
        # (100*64 + 64) + (64*32 + 32) + (32*16 + 16) + (16*10 + 10)
        # = 6464 + 2080 + 528 + 170 = 9242
        expected = 9242
        actual = variables['expected_params']
        
        if actual != expected:
            self.fail(f"Expected {expected} parameters, got {actual}")
            return False
        
        return True


# Define test sections
EXERCISE3_SECTIONS = {
    "Section 1: Understanding nn.Linear": [
        ("test_linear_layer_1", "Linear layer with 10 inputs and 5 outputs"),
        ("test_linear_layer_2", "Linear layer with 5 inputs and 3 outputs"),
        ("test_num_params_layer2", "Correct parameter count for linear_layer_2"),
    ],
    "Section 2: Activation Functions": [
        ("test_relu_activation", "ReLU activation function"),
        ("test_sigmoid_activation", "Sigmoid activation function"),
        ("test_tanh_activation", "Tanh activation function"),
        ("test_linear_output", "Linear layer output computation"),
        ("test_activated_output", "ReLU activation applied correctly"),
    ],
    "Section 3: Building Networks with nn.Sequential": [
        ("test_simple_mlp", "Simple 2-layer MLP structure"),
        ("test_deep_mlp", "Deep 4-layer MLP structure"),
        ("test_deep_mlp_params", "Correct parameter count for deep_mlp"),
    ],
    "Section 4: Forward Pass": [
        ("test_simple_output", "Forward pass through simple_mlp"),
        ("test_mixed_activation_mlp", "MLP with mixed activations"),
        ("test_mixed_output", "Forward pass through mixed_activation_mlp"),
    ],
    "Section 5: Understanding Parameter Counting": [
        ("test_count_parameters_function", "Parameter counting function"),
        ("test_large_mlp", "Large MLP structure"),
        ("test_expected_params", "Correct manual parameter calculation"),
    ],
}


if __name__ == "__main__":
    # Run tests
    validator = Exercise3Validator()
    runner = NotebookTestRunner("module1", 3)
    
    # You can run individual tests here for debugging
    print("Exercise 3 Validator ready for testing")