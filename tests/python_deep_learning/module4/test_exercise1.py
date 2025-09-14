import sys
import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple
import time

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner

class Exercise1Validator(TestValidator):
    """Validator for Module 4 Exercise 1: Model Resource Profiling"""
    
    def test_cpu_tensor(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test CPU tensor creation"""
        if not self.check_variable(namespace, 'cpu_tensor'):
            return False, "Variable 'cpu_tensor' not found"
        
        tensor = namespace['cpu_tensor']
        if not isinstance(tensor, torch.Tensor):
            return False, f"Expected torch.Tensor, got {type(tensor)}"
        
        if tensor.device.type != 'cpu':
            return False, f"Tensor should be on CPU, but is on {tensor.device}"
        
        if tensor.shape != (1000, 1000):
            return False, f"Expected shape (1000, 1000), got {tensor.shape}"
        
        return True, "CPU tensor created correctly"
    
    def test_gpu_tensor(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test GPU tensor creation (if available)"""
        if not self.check_variable(namespace, 'gpu_tensor'):
            return False, "Variable 'gpu_tensor' not found"
        
        if not torch.cuda.is_available():
            # If no GPU, accept None or message
            if namespace['gpu_tensor'] is None or isinstance(namespace['gpu_tensor'], str):
                return True, "No GPU available - correctly handled"
        
        tensor = namespace['gpu_tensor']
        if isinstance(tensor, torch.Tensor):
            if not tensor.is_cuda:
                return False, "Tensor should be on GPU when CUDA is available"
            if tensor.shape != (1000, 1000):
                return False, f"Expected shape (1000, 1000), got {tensor.shape}"
        
        return True, "GPU tensor handling is correct"
    
    def test_device_transfer_time(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test device transfer timing measurement"""
        if not self.check_variable(namespace, 'transfer_time'):
            return False, "Variable 'transfer_time' not found"
        
        time_val = namespace['transfer_time']
        if not isinstance(time_val, (float, int)):
            return False, f"Expected numeric time value, got {type(time_val)}"
        
        if time_val <= 0:
            return False, "Transfer time should be positive"
        
        return True, "Device transfer time measured correctly"
    
    def test_simple_model(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test SimpleModel class definition"""
        if not self.check_variable(namespace, 'SimpleModel'):
            return False, "Class 'SimpleModel' not found"
        
        model_class = namespace['SimpleModel']
        
        # Check if it's a class that inherits from nn.Module
        if not issubclass(model_class, nn.Module):
            return False, "SimpleModel should inherit from nn.Module"
        
        # Create instance and check structure
        try:
            model = model_class()
            
            # Check for required layers
            if not hasattr(model, 'fc1'):
                return False, "Model missing 'fc1' layer"
            if not hasattr(model, 'fc2'):
                return False, "Model missing 'fc2' layer"
            if not hasattr(model, 'fc3'):
                return False, "Model missing 'fc3' layer"
            
            # Test forward pass
            test_input = torch.randn(32, 784)
            output = model(test_input)
            
            if output.shape != (32, 10):
                return False, f"Expected output shape (32, 10), got {output.shape}"
            
        except Exception as e:
            return False, f"Error testing SimpleModel: {str(e)}"
        
        return True, "SimpleModel defined correctly"
    
    def test_cpu_training_time(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test CPU training time measurement"""
        if not self.check_variable(namespace, 'cpu_train_time'):
            return False, "Variable 'cpu_train_time' not found"
        
        time_val = namespace['cpu_train_time']
        if not isinstance(time_val, (float, int)):
            return False, f"Expected numeric time value, got {type(time_val)}"
        
        if time_val <= 0:
            return False, "CPU training time should be positive"
        
        return True, "CPU training time measured correctly"
    
    def test_gpu_training_time(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test GPU training time measurement"""
        if not self.check_variable(namespace, 'gpu_train_time'):
            return False, "Variable 'gpu_train_time' not found"
        
        time_val = namespace['gpu_train_time']
        
        # Accept None or string message if no GPU
        if not torch.cuda.is_available():
            if time_val is None or isinstance(time_val, str):
                return True, "No GPU available - correctly handled"
        
        if isinstance(time_val, (float, int)):
            if time_val <= 0:
                return False, "GPU training time should be positive"
        
        return True, "GPU training time measurement correct"
    
    def test_batch_size_comparison(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test batch size performance comparison"""
        if not self.check_variable(namespace, 'batch_times'):
            return False, "Variable 'batch_times' not found"
        
        batch_times = namespace['batch_times']
        if not isinstance(batch_times, dict):
            return False, f"Expected dict, got {type(batch_times)}"
        
        expected_sizes = [16, 32, 64, 128, 256]
        for size in expected_sizes:
            if size not in batch_times:
                return False, f"Missing batch size {size} in results"
            
            if not isinstance(batch_times[size], (float, int)):
                return False, f"Time for batch size {size} should be numeric"
            
            if batch_times[size] <= 0:
                return False, f"Time for batch size {size} should be positive"
        
        return True, "Batch size comparison completed correctly"
    
    def test_memory_usage(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test memory usage tracking"""
        if not self.check_variable(namespace, 'memory_before'):
            return False, "Variable 'memory_before' not found"
        if not self.check_variable(namespace, 'memory_after'):
            return False, "Variable 'memory_after' not found"
        
        mem_before = namespace['memory_before']
        mem_after = namespace['memory_after']
        
        if not isinstance(mem_before, (float, int)):
            return False, f"memory_before should be numeric, got {type(mem_before)}"
        
        if not isinstance(mem_after, (float, int)):
            return False, f"memory_after should be numeric, got {type(mem_after)}"
        
        if mem_before < 0 or mem_after < 0:
            return False, "Memory values should be non-negative"
        
        if mem_after <= mem_before:
            return False, "Memory after model creation should be greater than before"
        
        return True, "Memory usage tracked correctly"
    
    def test_operation_comparison(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test comparison of different operations"""
        if not self.check_variable(namespace, 'op_times'):
            return False, "Variable 'op_times' not found"
        
        op_times = namespace['op_times']
        if not isinstance(op_times, dict):
            return False, f"Expected dict, got {type(op_times)}"
        
        expected_ops = ['matmul', 'conv2d', 'relu', 'softmax']
        for op in expected_ops:
            if op not in op_times:
                return False, f"Missing operation '{op}' in results"
            
            if not isinstance(op_times[op], (float, int)):
                return False, f"Time for operation '{op}' should be numeric"
            
            if op_times[op] <= 0:
                return False, f"Time for operation '{op}' should be positive"
        
        return True, "Operation comparison completed correctly"
    
    def test_profiler_results(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test PyTorch profiler usage"""
        if not self.check_variable(namespace, 'profile_table'):
            return False, "Variable 'profile_table' not found"
        
        profile_table = namespace['profile_table']
        
        # Check if it's a string (profiler output)
        if not isinstance(profile_table, str):
            return False, f"Expected string profiler output, got {type(profile_table)}"
        
        # Check for key profiler output indicators
        if 'Self CPU' not in profile_table and 'CUDA' not in profile_table:
            return False, "Profile table should contain CPU or CUDA timing information"
        
        return True, "Profiler results generated correctly"
    
    def test_optimization_comparison(self, namespace: Dict[str, Any]) -> Tuple[bool, str]:
        """Test optimization technique comparison"""
        if not self.check_variable(namespace, 'optimization_results'):
            return False, "Variable 'optimization_results' not found"
        
        results = namespace['optimization_results']
        if not isinstance(results, dict):
            return False, f"Expected dict, got {type(results)}"
        
        expected_keys = ['baseline', 'optimized']
        for key in expected_keys:
            if key not in results:
                return False, f"Missing '{key}' in optimization results"
            
            if not isinstance(results[key], (float, int)):
                return False, f"Time for '{key}' should be numeric"
            
            if results[key] <= 0:
                return False, f"Time for '{key}' should be positive"
        
        # Optimized should generally be faster (but not always guaranteed)
        if results['optimized'] > results['baseline'] * 1.1:  # Allow 10% tolerance
            return False, "Optimized version should generally be faster than baseline"
        
        return True, "Optimization comparison completed correctly"

# Define sections and their tests
EXERCISE1_SECTIONS = {
    "Section 1: Device Management": [
        ("test_cpu_tensor", "CPU tensor creation"),
        ("test_gpu_tensor", "GPU tensor creation (if available)"),
        ("test_device_transfer_time", "Device transfer timing"),
    ],
    "Section 2: Model Training Performance": [
        ("test_simple_model", "SimpleModel class definition"),
        ("test_cpu_training_time", "CPU training time measurement"),
        ("test_gpu_training_time", "GPU training time measurement"),
    ],
    "Section 3: Batch Size Impact": [
        ("test_batch_size_comparison", "Batch size performance comparison"),
    ],
    "Section 4: Memory Profiling": [
        ("test_memory_usage", "Memory usage tracking"),
    ],
    "Section 5: Operation-Level Profiling": [
        ("test_operation_comparison", "Operation timing comparison"),
        ("test_profiler_results", "PyTorch profiler usage"),
    ],
    "Section 6: Optimization Techniques": [
        ("test_optimization_comparison", "Optimization technique comparison"),
    ],
}

if __name__ == "__main__":
    # For standalone testing
    validator = Exercise1Validator()
    print("Module 4 Exercise 1 Validator ready")