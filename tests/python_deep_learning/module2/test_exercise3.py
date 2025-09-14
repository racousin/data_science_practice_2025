import sys
import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, Any, List

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner


class Exercise3Validator(TestValidator):
    """Validator for Module 2 Exercise 3: Gradient Flow"""
    
    # Section 1: Vanishing Gradients with Sigmoid
    def test_deep_sigmoid_network(self, context: Dict[str, Any]) -> None:
        """Test creation of deep network with sigmoid activations"""
        self.check_variable(context, 'deep_sigmoid_network')
        network = context['deep_sigmoid_network']
        
        # Check it's a Sequential model
        assert isinstance(network, nn.Sequential), "Should be a nn.Sequential model"
        
        # Check number of layers (at least 10 layers total)
        num_layers = len(list(network.modules())) - 1  # Subtract the Sequential container
        assert num_layers >= 10, f"Network should have at least 10 layers, got {num_layers}"
        
        # Check for sigmoid activations
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in network.modules())
        assert has_sigmoid, "Network should contain Sigmoid activations"
    
    def test_sigmoid_gradients(self, context: Dict[str, Any]) -> None:
        """Test gradient computation through sigmoid network"""
        self.check_variable(context, 'sigmoid_gradients')
        gradients = context['sigmoid_gradients']
        
        assert isinstance(gradients, list), "Gradients should be a list"
        assert len(gradients) > 0, "Should have collected gradients"
        
        # Check for vanishing gradients (later gradients should be much smaller)
        if len(gradients) >= 2:
            first_grad_norm = torch.norm(gradients[0])
            last_grad_norm = torch.norm(gradients[-1])
            ratio = last_grad_norm / first_grad_norm
            assert ratio < 0.01 or last_grad_norm < 1e-5, "Should demonstrate vanishing gradients"
    
    def test_vanishing_gradient_ratio(self, context: Dict[str, Any]) -> None:
        """Test calculation of gradient vanishing ratio"""
        self.check_variable(context, 'vanishing_ratio')
        ratio = context['vanishing_ratio']
        
        assert isinstance(ratio, (float, torch.Tensor)), "Ratio should be a number"
        if isinstance(ratio, torch.Tensor):
            ratio = ratio.item()
        
        # With deep sigmoid network, ratio should be very small
        assert ratio < 0.01, f"Vanishing ratio should be < 0.01, got {ratio}"
    
    # Section 2: Vanishing Gradients with Tanh
    def test_deep_tanh_network(self, context: Dict[str, Any]) -> None:
        """Test creation of deep network with tanh activations"""
        self.check_variable(context, 'deep_tanh_network')
        network = context['deep_tanh_network']
        
        assert isinstance(network, nn.Sequential), "Should be a nn.Sequential model"
        
        # Check for tanh activations
        has_tanh = any(isinstance(m, nn.Tanh) for m in network.modules())
        assert has_tanh, "Network should contain Tanh activations"
        
        # Check depth
        num_layers = len(list(network.modules())) - 1
        assert num_layers >= 10, f"Network should have at least 10 layers, got {num_layers}"
    
    def test_tanh_gradients(self, context: Dict[str, Any]) -> None:
        """Test gradient computation through tanh network"""
        self.check_variable(context, 'tanh_gradients')
        gradients = context['tanh_gradients']
        
        assert isinstance(gradients, list), "Gradients should be a list"
        assert len(gradients) > 0, "Should have collected gradients"
        
        # Tanh should also show vanishing but less severe than sigmoid
        if len(gradients) >= 2:
            first_grad_norm = torch.norm(gradients[0])
            last_grad_norm = torch.norm(gradients[-1])
            ratio = last_grad_norm / first_grad_norm
            assert ratio < 0.1 or last_grad_norm < 1e-4, "Should demonstrate vanishing gradients with tanh"
    
    # Section 3: Exploding Gradients
    def test_unstable_network(self, context: Dict[str, Any]) -> None:
        """Test creation of network prone to exploding gradients"""
        self.check_variable(context, 'unstable_network')
        network = context['unstable_network']
        
        assert isinstance(network, nn.Sequential), "Should be a nn.Sequential model"
        
        # Check for large weight initialization
        for module in network.modules():
            if isinstance(module, nn.Linear):
                weight_std = module.weight.data.std().item()
                # At least some layers should have large weights
                if weight_std > 2.0:
                    return
        
        assert False, "Network should have layers with large weight initialization (std > 2.0)"
    
    def test_exploding_gradients(self, context: Dict[str, Any]) -> None:
        """Test detection of exploding gradients"""
        self.check_variable(context, 'exploding_gradients')
        gradients = context['exploding_gradients']
        
        assert isinstance(gradients, list), "Gradients should be a list"
        
        # Check for exploding gradients
        max_grad_norm = max(torch.norm(g).item() if torch.is_tensor(g) else abs(g) for g in gradients)
        assert max_grad_norm > 100 or any(torch.isnan(g).any() if torch.is_tensor(g) else np.isnan(g) for g in gradients), \
            f"Should demonstrate exploding gradients (max norm > 100), got max norm: {max_grad_norm}"
    
    def test_gradient_clipping(self, context: Dict[str, Any]) -> None:
        """Test implementation of gradient clipping"""
        self.check_variable(context, 'clipped_gradients')
        clipped = context['clipped_gradients']
        
        assert isinstance(clipped, list), "Clipped gradients should be a list"
        
        # Check that gradients are actually clipped
        max_norm = max(torch.norm(g).item() if torch.is_tensor(g) else abs(g) for g in clipped)
        assert max_norm <= 1.1, f"Gradients should be clipped to max norm of 1.0, got {max_norm}"
    
    # Section 4: Solutions - ReLU and Batch Normalization
    def test_relu_network(self, context: Dict[str, Any]) -> None:
        """Test network with ReLU activations"""
        self.check_variable(context, 'relu_network')
        network = context['relu_network']
        
        assert isinstance(network, nn.Sequential), "Should be a nn.Sequential model"
        
        # Check for ReLU activations
        has_relu = any(isinstance(m, nn.ReLU) for m in network.modules())
        assert has_relu, "Network should contain ReLU activations"
    
    def test_relu_gradients(self, context: Dict[str, Any]) -> None:
        """Test gradient flow through ReLU network"""
        self.check_variable(context, 'relu_gradients')
        gradients = context['relu_gradients']
        
        assert isinstance(gradients, list), "Gradients should be a list"
        
        # ReLU should maintain better gradient flow
        if len(gradients) >= 2:
            first_grad_norm = torch.norm(gradients[0]).item()
            last_grad_norm = torch.norm(gradients[-1]).item()
            if first_grad_norm > 0:
                ratio = last_grad_norm / first_grad_norm
                assert ratio > 0.01, f"ReLU should maintain better gradient flow, ratio: {ratio}"
    
    def test_batchnorm_network(self, context: Dict[str, Any]) -> None:
        """Test network with batch normalization"""
        self.check_variable(context, 'batchnorm_network')
        network = context['batchnorm_network']
        
        assert isinstance(network, nn.Sequential), "Should be a nn.Sequential model"
        
        # Check for BatchNorm layers
        has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in network.modules())
        assert has_batchnorm, "Network should contain BatchNorm layers"
    
    def test_batchnorm_gradients(self, context: Dict[str, Any]) -> None:
        """Test gradient flow through batch normalized network"""
        self.check_variable(context, 'batchnorm_gradients')
        gradients = context['batchnorm_gradients']
        
        assert isinstance(gradients, list), "Gradients should be a list"
        
        # BatchNorm should stabilize gradient flow
        gradient_stds = [torch.std(g).item() if torch.is_tensor(g) else 0 for g in gradients]
        # Check that gradient variance is relatively stable
        if len(gradient_stds) >= 2:
            std_ratio = max(gradient_stds) / (min(gradient_stds) + 1e-8)
            assert std_ratio < 100, f"BatchNorm should stabilize gradients, std ratio: {std_ratio}"
    
    # Section 5: Gradient Analysis
    def test_gradient_statistics(self, context: Dict[str, Any]) -> None:
        """Test computation of gradient statistics"""
        self.check_variable(context, 'gradient_stats')
        stats = context['gradient_stats']
        
        assert isinstance(stats, dict), "Should be a dictionary of statistics"
        
        required_keys = ['mean', 'std', 'min', 'max']
        for key in required_keys:
            assert key in stats, f"Statistics should include '{key}'"
            assert isinstance(stats[key], (float, torch.Tensor)), f"'{key}' should be a number"
    
    def test_gradient_histogram(self, context: Dict[str, Any]) -> None:
        """Test creation of gradient histogram data"""
        self.check_variable(context, 'gradient_histogram_data')
        hist_data = context['gradient_histogram_data']
        
        assert isinstance(hist_data, (list, np.ndarray, torch.Tensor)), \
            "Histogram data should be a list, array, or tensor"
        
        if isinstance(hist_data, list):
            assert len(hist_data) > 0, "Histogram data should not be empty"


# Define test sections for the exercise
EXERCISE3_SECTIONS = {
    "Section 1: Vanishing Gradients with Sigmoid": [
        ("test_deep_sigmoid_network", "Deep network with sigmoid activations"),
        ("test_sigmoid_gradients", "Gradient computation through sigmoid network"),
        ("test_vanishing_gradient_ratio", "Calculation of gradient vanishing ratio"),
    ],
    "Section 2: Vanishing Gradients with Tanh": [
        ("test_deep_tanh_network", "Deep network with tanh activations"),
        ("test_tanh_gradients", "Gradient computation through tanh network"),
    ],
    "Section 3: Exploding Gradients": [
        ("test_unstable_network", "Network prone to exploding gradients"),
        ("test_exploding_gradients", "Detection of exploding gradients"),
        ("test_gradient_clipping", "Implementation of gradient clipping"),
    ],
    "Section 4: Solutions - ReLU and Batch Normalization": [
        ("test_relu_network", "Network with ReLU activations"),
        ("test_relu_gradients", "Gradient flow through ReLU network"),
        ("test_batchnorm_network", "Network with batch normalization"),
        ("test_batchnorm_gradients", "Gradient flow through batch normalized network"),
    ],
    "Section 5: Gradient Analysis": [
        ("test_gradient_statistics", "Computation of gradient statistics"),
        ("test_gradient_histogram", "Creation of gradient histogram data"),
    ],
}


if __name__ == "__main__":
    # This allows running the tests directly
    validator = Exercise3Validator()
    runner = NotebookTestRunner("module2", 3)
    print("Exercise 3 Validator - Gradient Flow")