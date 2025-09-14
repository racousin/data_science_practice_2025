#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 1 - Exercise 1: Tensor Basics
Refactored to use shared test utilities and reduce duplication.
"""

import sys
import torch
import numpy as np
import pytest
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner


class Exercise1Validator(TestValidator):
    """Validator for Module 1 Exercise 1: Tensor Basics"""
    
    def test_tensor_zeros(self):
        """Test tensor_zeros creation"""
        tensor_zeros = self.check_variable('tensor_zeros', torch.Tensor)
        self.check_tensor_shape(tensor_zeros, (3, 3), "tensor_zeros")
        if not torch.all(tensor_zeros == 0):
            raise AssertionError("tensor_zeros should contain all zeros")
    
    def test_tensor_ones(self):
        """Test tensor_ones creation"""
        tensor_ones = self.check_variable('tensor_ones', torch.Tensor)
        self.check_tensor_shape(tensor_ones, (2, 4), "tensor_ones")
        if not torch.all(tensor_ones == 1):
            raise AssertionError("tensor_ones should contain all ones")
    
    def test_tensor_identity(self):
        """Test tensor_identity creation"""
        tensor_identity = self.check_variable('tensor_identity', torch.Tensor)
        self.check_tensor_shape(tensor_identity, (3, 3), "tensor_identity")
        expected = torch.eye(3)
        self.check_tensor_values(tensor_identity, expected, "tensor_identity")
    
    def test_tensor_random(self):
        """Test tensor_random creation"""
        tensor_random = self.check_variable('tensor_random', torch.Tensor)
        self.check_tensor_shape(tensor_random, (2, 3, 4), "tensor_random")
        if not torch.all((tensor_random >= 0) & (tensor_random <= 1)):
            raise AssertionError("tensor_random values should be between 0 and 1")
    
    def test_tensor_from_list(self):
        """Test tensor_from_list creation"""
        tensor_from_list = self.check_variable('tensor_from_list', torch.Tensor)
        self.check_tensor_shape(tensor_from_list, (2, 3), "tensor_from_list")
        expected = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.check_tensor_values(tensor_from_list, expected, "tensor_from_list")
    
    def test_tensor_range(self):
        """Test tensor_range creation"""
        tensor_range = self.check_variable('tensor_range', torch.Tensor)
        self.check_tensor_shape(tensor_range, (10,), "tensor_range")
        expected = torch.arange(10)
        self.check_tensor_values(tensor_range, expected, "tensor_range")
    
    def test_tensor_shape(self):
        """Test tensor_shape attribute"""
        # Ensure sample_tensor exists
        if 'sample_tensor' not in self.namespace:
            self.namespace['sample_tensor'] = torch.randn(3, 4, 5)
        
        sample_tensor = self.namespace['sample_tensor']
        tensor_shape = self.check_variable('tensor_shape')
        
        if tensor_shape != sample_tensor.shape:
            raise AssertionError(f"tensor_shape should be {sample_tensor.shape}, got {tensor_shape}")
    
    def test_tensor_dtype(self):
        """Test tensor_dtype attribute"""
        if 'sample_tensor' not in self.namespace:
            self.namespace['sample_tensor'] = torch.randn(3, 4, 5)
        
        sample_tensor = self.namespace['sample_tensor']
        tensor_dtype = self.check_variable('tensor_dtype')
        
        if tensor_dtype != sample_tensor.dtype:
            raise AssertionError(f"tensor_dtype should be {sample_tensor.dtype}, got {tensor_dtype}")
    
    def test_tensor_device(self):
        """Test tensor_device attribute"""
        if 'sample_tensor' not in self.namespace:
            self.namespace['sample_tensor'] = torch.randn(3, 4, 5)
        
        sample_tensor = self.namespace['sample_tensor']
        tensor_device = self.check_variable('tensor_device')
        
        if tensor_device != sample_tensor.device:
            raise AssertionError(f"tensor_device should be {sample_tensor.device}, got {tensor_device}")
    
    def test_tensor_ndim(self):
        """Test tensor_ndim attribute"""
        if 'sample_tensor' not in self.namespace:
            self.namespace['sample_tensor'] = torch.randn(3, 4, 5)
        
        sample_tensor = self.namespace['sample_tensor']
        tensor_ndim = self.check_variable('tensor_ndim')
        
        if tensor_ndim != sample_tensor.ndim:
            raise AssertionError(f"tensor_ndim should be {sample_tensor.ndim}, got {tensor_ndim}")
    
    def test_tensor_numel(self):
        """Test tensor_numel attribute"""
        if 'sample_tensor' not in self.namespace:
            self.namespace['sample_tensor'] = torch.randn(3, 4, 5)
        
        sample_tensor = self.namespace['sample_tensor']
        tensor_numel = self.check_variable('tensor_numel')
        
        if tensor_numel != sample_tensor.numel():
            raise AssertionError(f"tensor_numel should be {sample_tensor.numel()}, got {tensor_numel}")
    
    def test_indexing_element(self):
        """Test element indexing"""
        if 'tensor' not in self.namespace:
            self.namespace['tensor'] = torch.arange(24).reshape(4, 6)
        
        tensor = self.namespace['tensor']
        element = self.check_variable('element')
        expected = tensor[1, 3]
        
        if element != expected:
            raise AssertionError(f"element should be {expected}, got {element}")
    
    def test_indexing_row(self):
        """Test row indexing"""
        if 'tensor' not in self.namespace:
            self.namespace['tensor'] = torch.arange(24).reshape(4, 6)
        
        tensor = self.namespace['tensor']
        second_row = self.check_variable('second_row', torch.Tensor)
        expected = tensor[1]
        self.check_tensor_values(second_row, expected, "second_row")
    
    def test_indexing_column(self):
        """Test column indexing"""
        if 'tensor' not in self.namespace:
            self.namespace['tensor'] = torch.arange(24).reshape(4, 6)
        
        tensor = self.namespace['tensor']
        last_column = self.check_variable('last_column', torch.Tensor)
        expected = tensor[:, -1]
        self.check_tensor_values(last_column, expected, "last_column")
    
    def test_indexing_submatrix(self):
        """Test submatrix indexing"""
        if 'tensor' not in self.namespace:
            self.namespace['tensor'] = torch.arange(24).reshape(4, 6)
        
        tensor = self.namespace['tensor']
        submatrix = self.check_variable('submatrix', torch.Tensor)
        expected = tensor[:2, :2]
        self.check_tensor_values(submatrix, expected, "submatrix")
    
    def test_indexing_alternating(self):
        """Test alternating elements indexing"""
        if 'tensor' not in self.namespace:
            self.namespace['tensor'] = torch.arange(24).reshape(4, 6)
        
        tensor = self.namespace['tensor']
        alternating = self.check_variable('alternating_elements', torch.Tensor)
        expected = tensor[0, ::2]
        self.check_tensor_values(alternating, expected, "alternating_elements")
    
    def test_reshape_3x4(self):
        """Test reshape to 3x4"""
        if 'original' not in self.namespace:
            self.namespace['original'] = torch.arange(12)
        
        reshaped = self.check_variable('reshaped_3x4', torch.Tensor)
        self.check_tensor_shape(reshaped, (3, 4), "reshaped_3x4")
    
    def test_reshape_2x2x3(self):
        """Test reshape to 2x2x3"""
        if 'original' not in self.namespace:
            self.namespace['original'] = torch.arange(12)
        
        reshaped = self.check_variable('reshaped_2x2x3', torch.Tensor)
        self.check_tensor_shape(reshaped, (2, 2, 3), "reshaped_2x2x3")
    
    def test_flatten(self):
        """Test flatten operation"""
        if 'original' not in self.namespace:
            self.namespace['original'] = torch.arange(12)
        
        flattened = self.check_variable('flattened', torch.Tensor)
        self.check_tensor_shape(flattened, (12,), "flattened")
        expected = self.namespace['original']
        self.check_tensor_values(flattened, expected, "flattened")
    
    def test_unsqueeze(self):
        """Test unsqueeze operation"""
        unsqueezed = self.check_variable('unsqueezed', torch.Tensor)
        self.check_tensor_shape(unsqueezed, (1, 12), "unsqueezed")
    
    def test_squeeze(self):
        """Test squeeze operation"""
        squeezed = self.check_variable('squeezed', torch.Tensor)
        self.check_tensor_shape(squeezed, (3, 4), "squeezed")
    
    def test_dtype_float32(self):
        """Test float32 tensor"""
        tensor = self.check_variable('float32_tensor', torch.Tensor)
        self.check_tensor_dtype(tensor, torch.float32, "float32_tensor")
    
    def test_dtype_float64(self):
        """Test float64 tensor"""
        tensor = self.check_variable('float64_tensor', torch.Tensor)
        self.check_tensor_dtype(tensor, torch.float64, "float64_tensor")
    
    def test_dtype_int_to_float(self):
        """Test int to float conversion"""
        int_tensor = self.check_variable('int_tensor', torch.Tensor)
        if int_tensor.dtype not in [torch.int32, torch.int64]:
            raise AssertionError(f"int_tensor should have integer dtype, got {int_tensor.dtype}")
        
        float_tensor = self.check_variable('int_to_float', torch.Tensor)
        self.check_tensor_dtype(float_tensor, torch.float32, "int_to_float")
    
    def test_dtype_bool(self):
        """Test boolean tensor"""
        bool_tensor = self.check_variable('bool_tensor', torch.Tensor)
        self.check_tensor_dtype(bool_tensor, torch.bool, "bool_tensor")
        expected = torch.tensor([False, False, False, True, True])
        self.check_tensor_values(bool_tensor, expected, "bool_tensor")
    
    def test_numpy_to_tensor(self):
        """Test NumPy to tensor conversion"""
        tensor = self.check_variable('tensor_from_numpy', torch.Tensor)
        self.check_tensor_shape(tensor, (2, 3), "tensor_from_numpy")
        
        if 'numpy_array' in self.namespace:
            expected = torch.from_numpy(self.namespace['numpy_array'])
            self.check_tensor_values(tensor, expected, "tensor_from_numpy")
    
    def test_tensor_to_numpy(self):
        """Test tensor to NumPy conversion"""
        numpy_array = self.check_variable('numpy_from_tensor', np.ndarray)
        if numpy_array.shape != (2, 3):
            raise AssertionError(f"numpy_from_tensor should have shape (2, 3), got {numpy_array.shape}")
    
    def test_shared_memory(self):
        """Test shared memory between NumPy and tensor"""
        shared_tensor = self.check_variable('shared_tensor', torch.Tensor)
        
        if 'shared_numpy' in self.namespace:
            shared_numpy = self.namespace['shared_numpy']
            # Test memory sharing
            original_value = shared_numpy[0, 0]
            shared_numpy[0, 0] = 999
            
            if shared_tensor[0, 0] != 999:
                raise AssertionError("shared_tensor should reflect changes in shared_numpy (memory sharing)")
            
            # Restore original value
            shared_numpy[0, 0] = original_value


# Define test sections for notebook testing
EXERCISE1_SECTIONS = {
    "Section 1: Tensor Creation": [
        ('test_tensor_zeros', "Create 3x3 tensor of zeros"),
        ('test_tensor_ones', "Create 2x4 tensor of ones"),
        ('test_tensor_identity', "Create 3x3 identity matrix"),
        ('test_tensor_random', "Create random tensor (2,3,4)"),
        ('test_tensor_from_list', "Create tensor from list"),
        ('test_tensor_range', "Create tensor with range 0-9"),
    ],
    "Section 3: Tensor Attributes": [
        ('test_tensor_shape', "Get tensor shape"),
        ('test_tensor_dtype', "Get tensor dtype"),
        ('test_tensor_device', "Get tensor device"),
        ('test_tensor_ndim', "Get tensor dimensions"),
        ('test_tensor_numel', "Get number of elements"),
    ],
    "Section 4: Indexing and Slicing": [
        ('test_indexing_element', "Access element at (1,3)"),
        ('test_indexing_row', "Get second row"),
        ('test_indexing_column', "Get last column"),
        ('test_indexing_submatrix', "Get 2x2 submatrix"),
        ('test_indexing_alternating', "Get alternating elements"),
    ],
    "Section 5: Tensor Reshaping": [
        ('test_reshape_3x4', "Reshape to 3x4"),
        ('test_reshape_2x2x3', "Reshape to 2x2x3"),
        ('test_flatten', "Flatten tensor"),
        ('test_unsqueeze', "Add dimension with unsqueeze"),
        ('test_squeeze', "Remove single dimensions"),
    ],
    "Section 6: Data Types": [
        ('test_dtype_float32', "Create float32 tensor"),
        ('test_dtype_float64', "Convert to float64"),
        ('test_dtype_int_to_float', "Convert int to float"),
        ('test_dtype_bool', "Create boolean tensor"),
    ],
    "Section 7: NumPy Interoperability": [
        ('test_numpy_to_tensor', "Convert NumPy to tensor"),
        ('test_tensor_to_numpy', "Convert tensor to NumPy"),
        ('test_shared_memory', "Test shared memory"),
    ],
}


def create_notebook_tests(namespace: Dict[str, Any]) -> NotebookTestRunner:
    """
    Create notebook test runner for Exercise 1.
    
    Args:
        namespace: The notebook's namespace (typically locals())
        
    Returns:
        Configured NotebookTestRunner instance
    """
    runner = NotebookTestRunner("module1", 1)
    validator = Exercise1Validator(namespace)
    
    return runner, validator


# Pytest test class for traditional testing
class TestExercise1:
    """Test cases for Exercise 1: Tensor Basics (pytest compatible)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a namespace with all required variables for testing
        self.namespace = {
            'tensor_zeros': torch.zeros(3, 3),
            'tensor_ones': torch.ones(2, 4),
            'tensor_identity': torch.eye(3),
            'tensor_random': torch.rand(2, 3, 4),
            'tensor_from_list': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'tensor_range': torch.arange(10),
            'sample_tensor': torch.randn(3, 4, 5),
            'tensor': torch.arange(24).reshape(4, 6),
            'original': torch.arange(12),
        }
        
        # Add attributes
        self.namespace['tensor_shape'] = self.namespace['sample_tensor'].shape
        self.namespace['tensor_dtype'] = self.namespace['sample_tensor'].dtype
        self.namespace['tensor_device'] = self.namespace['sample_tensor'].device
        self.namespace['tensor_ndim'] = self.namespace['sample_tensor'].ndim
        self.namespace['tensor_numel'] = self.namespace['sample_tensor'].numel()
        
        # Add indexing results
        tensor = self.namespace['tensor']
        self.namespace['element'] = tensor[1, 3]
        self.namespace['second_row'] = tensor[1]
        self.namespace['last_column'] = tensor[:, -1]
        self.namespace['submatrix'] = tensor[:2, :2]
        self.namespace['alternating_elements'] = tensor[0, ::2]
        
        # Add reshaping results
        original = self.namespace['original']
        self.namespace['reshaped_3x4'] = original.reshape(3, 4)
        self.namespace['reshaped_2x2x3'] = original.reshape(2, 2, 3)
        self.namespace['flattened'] = self.namespace['reshaped_2x2x3'].flatten()
        self.namespace['unsqueezed'] = original.unsqueeze(0)
        self.namespace['squeezed'] = torch.randn(1, 3, 1, 4).squeeze()
        
        # Add dtype results
        self.namespace['float32_tensor'] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        self.namespace['float64_tensor'] = self.namespace['float32_tensor'].double()
        self.namespace['int_tensor'] = torch.tensor([1, 2, 3])
        self.namespace['int_to_float'] = self.namespace['int_tensor'].float()
        self.namespace['bool_tensor'] = torch.tensor([1, 2, 3, 4, 5]) > 3
        
        # Add numpy interop
        self.namespace['numpy_array'] = np.array([[1, 2, 3], [4, 5, 6]])
        self.namespace['tensor_from_numpy'] = torch.from_numpy(self.namespace['numpy_array'])
        self.namespace['pytorch_tensor'] = torch.randn(2, 3)
        self.namespace['numpy_from_tensor'] = self.namespace['pytorch_tensor'].numpy()
        self.namespace['shared_numpy'] = np.ones((2, 2))
        self.namespace['shared_tensor'] = torch.from_numpy(self.namespace['shared_numpy'])
        
        self.validator = Exercise1Validator(self.namespace)
    
    def test_all_sections(self):
        """Run all tests for all sections"""
        for section_name, tests in EXERCISE1_SECTIONS.items():
            for test_method_name, test_description in tests:
                test_method = getattr(self.validator, test_method_name)
                test_method()  # Will raise if test fails


def run_tests():
    """Run all tests for Exercise 1"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()