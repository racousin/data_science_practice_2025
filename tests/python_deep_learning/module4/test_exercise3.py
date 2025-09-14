import sys
import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple
import time

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner

class Exercise3Validator(TestValidator):
    """Validator for Module 4 Exercise 3: Performance Optimization Techniques"""
    
    def test_simple_cnn(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test SimpleCNN model definition"""
        if not self.check_variable(namespace, 'SimpleCNN'):
            return False, "Class 'SimpleCNN' not found"
        
        model_class = namespace['SimpleCNN']
        
        # Check if it's a class that inherits from nn.Module
        if not issubclass(model_class, nn.Module):
            return False, "SimpleCNN should inherit from nn.Module"
        
        # Create instance and check structure
        try:
            model = model_class()
            
            # Check for required layers
            if not hasattr(model, 'conv1'):
                return False, "Model missing 'conv1' layer"
            if not hasattr(model, 'conv2'):
                return False, "Model missing 'conv2' layer"
            if not hasattr(model, 'fc1'):
                return False, "Model missing 'fc1' layer"
            if not hasattr(model, 'fc2'):
                return False, "Model missing 'fc2' layer"
            
            # Test forward pass
            test_input = torch.randn(8, 1, 28, 28)
            output = model(test_input)
            
            if output.shape != (8, 10):
                return False, f"Expected output shape (8, 10), got {output.shape}"
            
        except Exception as e:
            return False, f"Error testing SimpleCNN: {str(e)}"
        
        return True, "SimpleCNN defined correctly"
    
    def test_baseline_inference_time(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test baseline inference time measurement"""
        if not self.check_variable(namespace, 'baseline_time'):
            return False, "Variable 'baseline_time' not found"
        
        time_val = namespace['baseline_time']
        if not isinstance(time_val, (float, int)):
            return False, f"Expected numeric time value, got {type(time_val)}"
        
        if time_val <= 0:
            return False, "Baseline inference time should be positive"
        
        return True, "Baseline inference time measured correctly"
    
    def test_compiled_model(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test torch.compile() model creation"""
        if not self.check_variable(namespace, 'compiled_model'):
            return False, "Variable 'compiled_model' not found"
        
        model = namespace['compiled_model']
        
        # Check if model can perform inference
        try:
            test_input = torch.randn(8, 1, 28, 28)
            with torch.no_grad():
                output = model(test_input)
            
            if output.shape != (8, 10):
                return False, f"Compiled model output shape incorrect: {output.shape}"
        except Exception as e:
            return False, f"Error testing compiled model: {str(e)}"
        
        return True, "Compiled model created correctly"
    
    def test_compiled_inference_time(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test compiled model inference time"""
        if not self.check_variable(namespace, 'compiled_time'):
            return False, "Variable 'compiled_time' not found"
        
        time_val = namespace['compiled_time']
        if not isinstance(time_val, (float, int)):
            return False, f"Expected numeric time value, got {type(time_val)}"
        
        if time_val <= 0:
            return False, "Compiled inference time should be positive"
        
        return True, "Compiled inference time measured correctly"
    
    def test_compilation_speedup(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test compilation speedup calculation"""
        if not self.check_variable(namespace, 'compile_speedup'):
            return False, "Variable 'compile_speedup' not found"
        
        speedup = namespace['compile_speedup']
        if not isinstance(speedup, (float, int)):
            return False, f"Expected numeric speedup value, got {type(speedup)}"
        
        if speedup <= 0:
            return False, "Speedup should be positive"
        
        return True, "Compilation speedup calculated correctly"
    
    def test_pruning_function(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test pruning function implementation"""
        if not self.check_variable(namespace, 'apply_pruning'):
            return False, "Function 'apply_pruning' not found"
        
        func = namespace['apply_pruning']
        if not callable(func):
            return False, "apply_pruning should be a callable function"
        
        # Test the function
        try:
            test_model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10)
            )
            
            # Count parameters before
            params_before = sum(p.numel() for p in test_model.parameters())
            
            # Apply pruning
            pruned_model = func(test_model, 0.3)
            
            # Should return a model
            if not isinstance(pruned_model, nn.Module):
                return False, "apply_pruning should return a nn.Module"
            
        except Exception as e:
            return False, f"Error testing apply_pruning: {str(e)}"
        
        return True, "Pruning function implemented correctly"
    
    def test_pruned_model(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test pruned model creation"""
        if not self.check_variable(namespace, 'pruned_model'):
            return False, "Variable 'pruned_model' not found"
        
        model = namespace['pruned_model']
        
        # Check if model can perform inference
        try:
            test_input = torch.randn(8, 1, 28, 28)
            with torch.no_grad():
                output = model(test_input)
            
            if output.shape != (8, 10):
                return False, f"Pruned model output shape incorrect: {output.shape}"
        except Exception as e:
            return False, f"Error testing pruned model: {str(e)}"
        
        return True, "Pruned model created correctly"
    
    def test_sparsity_calculation(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test sparsity calculation"""
        if not self.check_variable(namespace, 'sparsity'):
            return False, "Variable 'sparsity' not found"
        
        sparsity = namespace['sparsity']
        if not isinstance(sparsity, (float, int)):
            return False, f"Expected numeric sparsity value, got {type(sparsity)}"
        
        if sparsity < 0 or sparsity > 100:
            return False, f"Sparsity should be between 0 and 100, got {sparsity}"
        
        # Should have some sparsity after pruning
        if sparsity < 10:
            return False, "Sparsity seems too low after pruning"
        
        return True, "Sparsity calculated correctly"
    
    def test_pruned_inference_time(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test pruned model inference time"""
        if not self.check_variable(namespace, 'pruned_time'):
            return False, "Variable 'pruned_time' not found"
        
        time_val = namespace['pruned_time']
        if not isinstance(time_val, (float, int)):
            return False, f"Expected numeric time value, got {type(time_val)}"
        
        if time_val <= 0:
            return False, "Pruned inference time should be positive"
        
        return True, "Pruned inference time measured correctly"
    
    def test_mixed_precision_context(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test mixed precision training setup"""
        if not self.check_variable(namespace, 'mixed_precision_time'):
            return False, "Variable 'mixed_precision_time' not found"
        
        time_val = namespace['mixed_precision_time']
        
        # Check if it's numeric or None (in case of no GPU)
        if time_val is not None:
            if not isinstance(time_val, (float, int)):
                return False, f"Expected numeric time value or None, got {type(time_val)}"
            
            if isinstance(time_val, (float, int)) and time_val <= 0:
                return False, "Mixed precision time should be positive"
        
        return True, "Mixed precision training tested correctly"
    
    def test_fp32_training_time(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test FP32 training time"""
        if not self.check_variable(namespace, 'fp32_time'):
            return False, "Variable 'fp32_time' not found"
        
        time_val = namespace['fp32_time']
        if not isinstance(time_val, (float, int)):
            return False, f"Expected numeric time value, got {type(time_val)}"
        
        if time_val <= 0:
            return False, "FP32 training time should be positive"
        
        return True, "FP32 training time measured correctly"
    
    def test_memory_comparison(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test memory usage comparison"""
        if not self.check_variable(namespace, 'memory_comparison'):
            return False, "Variable 'memory_comparison' not found"
        
        mem_comp = namespace['memory_comparison']
        if not isinstance(mem_comp, dict):
            return False, f"Expected dict, got {type(mem_comp)}"
        
        expected_keys = ['fp32', 'fp16']
        for key in expected_keys:
            if key not in mem_comp:
                # FP16 might be None if no GPU
                if key == 'fp16' and 'fp16' in mem_comp and mem_comp['fp16'] is None:
                    continue
                return False, f"Missing '{key}' in memory comparison"
            
            if mem_comp[key] is not None:
                if not isinstance(mem_comp[key], (float, int)):
                    return False, f"Memory for '{key}' should be numeric"
                
                if mem_comp[key] < 0:
                    return False, f"Memory for '{key}' should be non-negative"
        
        return True, "Memory comparison completed correctly"
    
    def test_optimization_summary(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test optimization summary results"""
        if not self.check_variable(namespace, 'optimization_summary'):
            return False, "Variable 'optimization_summary' not found"
        
        summary = namespace['optimization_summary']
        if not isinstance(summary, dict):
            return False, f"Expected dict, got {type(summary)}"
        
        expected_techniques = ['baseline', 'compiled', 'pruned']
        for technique in expected_techniques:
            if technique not in summary:
                return False, f"Missing '{technique}' in optimization summary"
            
            if not isinstance(summary[technique], (float, int)):
                return False, f"Time for '{technique}' should be numeric"
            
            if summary[technique] <= 0:
                return False, f"Time for '{technique}' should be positive"
        
        # Mixed precision might be None if no GPU
        if 'mixed_precision' in summary:
            if summary['mixed_precision'] is not None:
                if not isinstance(summary['mixed_precision'], (float, int)):
                    return False, "Time for 'mixed_precision' should be numeric or None"
                
                if summary['mixed_precision'] <= 0:
                    return False, "Time for 'mixed_precision' should be positive"
        
        return True, "Optimization summary created correctly"
    
    def test_combined_optimizations(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test combined optimization techniques"""
        if not self.check_variable(namespace, 'combined_model'):
            return False, "Variable 'combined_model' not found"
        
        model = namespace['combined_model']
        
        # Check if model can perform inference
        try:
            test_input = torch.randn(8, 1, 28, 28)
            with torch.no_grad():
                output = model(test_input)
            
            if output.shape != (8, 10):
                return False, f"Combined model output shape incorrect: {output.shape}"
        except Exception as e:
            return False, f"Error testing combined model: {str(e)}"
        
        return True, "Combined optimizations model created correctly"
    
    def test_combined_performance(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test combined optimizations performance"""
        if not self.check_variable(namespace, 'combined_time'):
            return False, "Variable 'combined_time' not found"
        
        time_val = namespace['combined_time']
        if not isinstance(time_val, (float, int)):
            return False, f"Expected numeric time value, got {type(time_val)}"
        
        if time_val <= 0:
            return False, "Combined optimization time should be positive"
        
        # Check if improvement exists
        if not self.check_variable(namespace, 'total_speedup'):
            return False, "Variable 'total_speedup' not found"
        
        speedup = namespace['total_speedup']
        if not isinstance(speedup, (float, int)):
            return False, f"Expected numeric speedup value, got {type(speedup)}"
        
        return True, "Combined optimizations performance measured correctly"

# Define sections and their tests
EXERCISE3_SECTIONS = {
    "Section 1: Model Compilation with torch.compile": [
        ("test_simple_cnn", "SimpleCNN model definition"),
        ("test_baseline_inference_time", "Baseline inference time measurement"),
        ("test_compiled_model", "Compiled model creation"),
        ("test_compiled_inference_time", "Compiled inference time measurement"),
        ("test_compilation_speedup", "Compilation speedup calculation"),
    ],
    "Section 2: Model Pruning": [
        ("test_pruning_function", "Pruning function implementation"),
        ("test_pruned_model", "Pruned model creation"),
        ("test_sparsity_calculation", "Sparsity calculation"),
        ("test_pruned_inference_time", "Pruned inference time measurement"),
    ],
    "Section 3: Mixed Precision Training": [
        ("test_mixed_precision_context", "Mixed precision training setup"),
        ("test_fp32_training_time", "FP32 training time measurement"),
        ("test_memory_comparison", "Memory usage comparison"),
    ],
    "Section 4: Optimization Comparison": [
        ("test_optimization_summary", "Optimization summary results"),
    ],
    "Section 5: Combined Optimizations": [
        ("test_combined_optimizations", "Combined optimization techniques"),
        ("test_combined_performance", "Combined optimizations performance"),
    ],
}

if __name__ == "__main__":
    # For standalone testing
    validator = Exercise3Validator()
    print("Module 4 Exercise 3 Validator ready")