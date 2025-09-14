#!/usr/bin/env python3
"""
Test suite for Python Deep Learning Module 2 - Exercise 1: Autograd Exploration
"""

import sys
import torch
import numpy as np
import pytest
from typing import Dict, Any

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner


class Exercise1Validator(TestValidator):
    """Validator for Module 2 Exercise 1: Autograd Exploration"""
    
    def test_x(self, namespace: Dict[str, Any]) -> None:
        """Test basic tensor with requires_grad"""
        self.check_variable(namespace, 'x', "x tensor with requires_grad=True")
        x = namespace['x']
        assert x.requires_grad, "x should have requires_grad=True"
        assert x.item() == 2.0, "x should have value 2.0"
    
    def test_y(self, namespace: Dict[str, Any]) -> None:
        """Test function computation y = x^2 + 3*x + 1"""
        self.check_variable(namespace, 'y', "y = x^2 + 3*x + 1")
        y = namespace['y']
        expected_y = 2.0**2 + 3*2.0 + 1  # 4 + 6 + 1 = 11
        assert torch.isclose(y, torch.tensor(expected_y)), f"y should equal {expected_y}, got {y}"
    
    def test_x_grad(self, namespace: Dict[str, Any]) -> None:
        """Test gradient computation for x"""
        self.check_variable(namespace, 'x', "x tensor")
        x = namespace['x']
        if not hasattr(x, 'grad') or x.grad is None:
            raise AssertionError("x.grad not found or is None. Please call y.backward() to compute gradients")
        expected_grad = 2*2.0 + 3  # 7
        assert torch.isclose(x.grad, torch.tensor(expected_grad)), f"x.grad should be {expected_grad}, got {x.grad}"
    
    def test_x1(self, namespace: Dict[str, Any]) -> None:
        """Test x1 tensor creation"""
        self.check_variable(namespace, 'x1', "x1 tensor with requires_grad=True")
        x1 = namespace['x1']
        assert x1.requires_grad, "x1 should have requires_grad=True"
        assert x1.item() == 1.0, "x1 should have value 1.0"
    
    def test_x2(self, namespace: Dict[str, Any]) -> None:
        """Test x2 tensor creation"""
        self.check_variable(namespace, 'x2', "x2 tensor with requires_grad=True")
        x2 = namespace['x2']
        assert x2.requires_grad, "x2 should have requires_grad=True"
        assert x2.item() == 2.0, "x2 should have value 2.0"
    
    def test_z(self, namespace: Dict[str, Any]) -> None:
        """Test multivariable function z = x1^2 + x2^3 + x1*x2"""
        self.check_variable(namespace, 'z', "z = x1^2 + x2^3 + x1*x2")
        z = namespace['z']
        x1, x2 = namespace['x1'], namespace['x2']
        expected_z = x1**2 + x2**3 + x1*x2
        assert torch.isclose(z, expected_z), f"z should equal x1^2 + x2^3 + x1*x2"
    
    def test_x1_grad(self, namespace: Dict[str, Any]) -> None:
        """Test gradient computation for x1"""
        self.check_variable(namespace, 'x1', "x1 tensor")
        x1, x2 = namespace['x1'], namespace['x2']
        if x1.grad is None:
            raise AssertionError("x1.grad not computed. Please call z.backward()")
        expected_x1_grad = 2*x1.item() + x2.item()  # 2*1 + 2 = 4
        assert torch.isclose(x1.grad, torch.tensor(expected_x1_grad)), f"x1.grad should be {expected_x1_grad}, got {x1.grad}"
    
    def test_x2_grad(self, namespace: Dict[str, Any]) -> None:
        """Test gradient computation for x2"""
        self.check_variable(namespace, 'x2', "x2 tensor")
        x1, x2 = namespace['x1'], namespace['x2']
        if x2.grad is None:
            raise AssertionError("x2.grad not computed. Please call z.backward()")
        expected_x2_grad = 3*(x2.item()**2) + x1.item()  # 3*4 + 1 = 13
        assert torch.isclose(x2.grad, torch.tensor(expected_x2_grad)), f"x2.grad should be {expected_x2_grad}, got {x2.grad}"
    
    def test_vec_x(self, namespace: Dict[str, Any]) -> None:
        """Test vector creation"""
        self.check_variable(namespace, 'vec_x', "vec_x tensor with requires_grad=True")
        vec_x = namespace['vec_x']
        assert vec_x.requires_grad, "vec_x should have requires_grad=True"
        self.check_tensor_shape(vec_x, (3,), "vec_x")
        expected_values = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(vec_x, expected_values), f"vec_x should be [1.0, 2.0, 3.0], got {vec_x}"
    
    def test_vec_sum(self, namespace: Dict[str, Any]) -> None:
        """Test vector loss computation"""
        self.check_variable(namespace, 'vec_sum', "vec_sum = sum of squares")
        vec_sum = namespace['vec_sum']
        vec_x = namespace['vec_x']
        expected_loss = torch.sum(vec_x**2)
        assert torch.isclose(vec_sum, expected_loss), "vec_sum should be sum of squares of vec_x"
    
    def test_vec_x_grad(self, namespace: Dict[str, Any]) -> None:
        """Test vector gradient computation"""
        self.check_variable(namespace, 'vec_x', "vec_x tensor")
        vec_x = namespace['vec_x']
        if vec_x.grad is None:
            raise AssertionError("vec_x.grad not computed. Please call vec_sum.backward()")
        expected_grad = 2 * vec_x.detach()
        assert torch.allclose(vec_x.grad, expected_grad), f"vec_x.grad should be 2*vec_x = {expected_grad}, got {vec_x.grad}"
    
    def test_mat_A(self, namespace: Dict[str, Any]) -> None:
        """Test matrix creation"""
        self.check_variable(namespace, 'mat_A', "mat_A tensor with requires_grad=True")
        mat_A = namespace['mat_A']
        assert mat_A.requires_grad, "mat_A should have requires_grad=True"
        self.check_tensor_shape(mat_A, (2, 2), "mat_A")
        expected_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(mat_A, expected_values), f"mat_A should be [[1, 2], [3, 4]], got {mat_A}"
    
    def test_mat_sum(self, namespace: Dict[str, Any]) -> None:
        """Test matrix loss computation"""
        self.check_variable(namespace, 'mat_sum', "mat_sum = sum of squares")
        mat_sum = namespace['mat_sum']
        mat_A = namespace['mat_A']
        expected_loss = torch.sum(mat_A**2)
        assert torch.isclose(mat_sum, expected_loss), "mat_sum should be sum of squares of mat_A"
    
    def test_mat_A_grad(self, namespace: Dict[str, Any]) -> None:
        """Test matrix gradient computation"""
        self.check_variable(namespace, 'mat_A', "mat_A tensor")
        mat_A = namespace['mat_A']
        if mat_A.grad is None:
            raise AssertionError("mat_A.grad not computed. Please call mat_sum.backward()")
        expected_grad = 2 * mat_A.detach()
        assert torch.allclose(mat_A.grad, expected_grad), f"mat_A.grad should be 2*mat_A"
    
    def test_graph_x(self, namespace: Dict[str, Any]) -> None:
        """Test computational graph x variable"""
        self.check_variable(namespace, 'graph_x', "graph_x tensor with requires_grad=True")
        graph_x = namespace['graph_x']
        assert graph_x.requires_grad, "graph_x should have requires_grad=True"
        assert graph_x.item() == 2.0, "graph_x should have value 2.0"
    
    def test_graph_y(self, namespace: Dict[str, Any]) -> None:
        """Test computational graph y = x^2"""
        self.check_variable(namespace, 'graph_y', "graph_y = graph_x^2")
        graph_y = namespace['graph_y']
        graph_x = namespace['graph_x']
        expected_y = graph_x**2
        assert torch.isclose(graph_y, expected_y), f"graph_y should be graph_x^2 = {expected_y}, got {graph_y}"
    
    def test_graph_z(self, namespace: Dict[str, Any]) -> None:
        """Test computational graph z = 3*y + 1"""
        self.check_variable(namespace, 'graph_z', "graph_z = 3*graph_y + 1")
        graph_z = namespace['graph_z']
        graph_y = namespace['graph_y']
        expected_z = 3*graph_y + 1
        assert torch.isclose(graph_z, expected_z), f"graph_z should be 3*graph_y + 1 = {expected_z}, got {graph_z}"
    
    def test_graph_w(self, namespace: Dict[str, Any]) -> None:
        """Test computational graph w = z^2"""
        self.check_variable(namespace, 'graph_w', "graph_w = graph_z^2")
        graph_w = namespace['graph_w']
        graph_z = namespace['graph_z']
        expected_w = graph_z**2
        assert torch.isclose(graph_w, expected_w), f"graph_w should be graph_z^2 = {expected_w}, got {graph_w}"
    
    def test_graph_x_grad(self, namespace: Dict[str, Any]) -> None:
        """Test computational graph gradient"""
        self.check_variable(namespace, 'graph_x', "graph_x tensor")
        graph_x = namespace['graph_x']
        if graph_x.grad is None:
            raise AssertionError("graph_x.grad not computed. Please call graph_w.backward()")
        # For x=2: y=4, z=13, w=169, dw/dx = 12*z*x = 12*13*2 = 312
        expected_grad = 12 * 13 * 2
        assert torch.isclose(graph_x.grad, torch.tensor(float(expected_grad))), f"graph_x.grad should be {expected_grad}, got {graph_x.grad}"
    
    def test_no_grad_result(self, namespace: Dict[str, Any]) -> None:
        """Test no_grad context result"""
        self.check_variable(namespace, 'no_grad_result', "no_grad_result from torch.no_grad() context")
        no_grad_result = namespace['no_grad_result']
        assert not no_grad_result.requires_grad, "no_grad_result should not require gradients"
        expected_value = 3.0**2 + 2*3.0  # 9 + 6 = 15
        assert torch.isclose(no_grad_result, torch.tensor(expected_value)), f"no_grad_result should be {expected_value}, got {no_grad_result}"
    
    def test_detached_result(self, namespace: Dict[str, Any]) -> None:
        """Test detached tensor"""
        self.check_variable(namespace, 'detached_result', "detached_result from tensor.detach()")
        detached_result = namespace['detached_result']
        assert not detached_result.requires_grad, "detached_result should not require gradients after detach"
        # Should have value y = x^3 + x = 3^3 + 3 = 27 + 3 = 30
        expected_value = 3.0**3 + 3.0  # 30
        assert torch.isclose(detached_result, torch.tensor(expected_value)), f"detached_result should be {expected_value}, got {detached_result}"
    
    def test_second_derivative(self, namespace: Dict[str, Any]) -> None:
        """Test second derivative computation"""
        self.check_variable(namespace, 'second_derivative', "second_derivative computation")
        second_derivative = namespace['second_derivative']
        # For f(x) = x^4, f''(x) = 12x^2. At x=2, f''(2) = 12*4 = 48
        expected_second_deriv = 48.0
        assert torch.isclose(second_derivative, torch.tensor(expected_second_deriv)), f"Second derivative should be {expected_second_deriv}, got {second_derivative}"


# Define section mappings for the notebook
EXERCISE1_SECTIONS = {
    "Section 1: Basic Autograd Operations": [
        ("test_x", "Create x = 2.0 with requires_grad=True"),
        ("test_y", "Compute y = x^2 + 3*x + 1"),
        ("test_x_grad", "Compute gradient dy/dx using backward()"),
    ],
    "Section 2: Multivariable Gradients": [
        ("test_x1", "Create x1 = 1.0 with requires_grad=True"),
        ("test_x2", "Create x2 = 2.0 with requires_grad=True"),
        ("test_z", "Compute z = x1^2 + x2^3 + x1*x2"),
        ("test_x1_grad", "Compute partial derivative ∂z/∂x1"),
        ("test_x2_grad", "Compute partial derivative ∂z/∂x2"),
    ],
    "Section 3: Vector and Matrix Gradients": [
        ("test_vec_x", "Create vector [1.0, 2.0, 3.0] with requires_grad=True"),
        ("test_vec_sum", "Compute vec_sum = sum of squares"),
        ("test_vec_x_grad", "Compute gradient for vector"),
        ("test_mat_A", "Create 2x2 matrix [[1, 2], [3, 4]] with requires_grad=True"),
        ("test_mat_sum", "Compute mat_sum = sum of squares"),
        ("test_mat_A_grad", "Compute gradient for matrix"),
    ],
    "Section 4: Computational Graph and Chain Rule": [
        ("test_graph_x", "Create graph_x = 2.0 with requires_grad=True"),
        ("test_graph_y", "Compute graph_y = graph_x^2"),
        ("test_graph_z", "Compute graph_z = 3*graph_y + 1"),
        ("test_graph_w", "Compute graph_w = graph_z^2"),
        ("test_graph_x_grad", "Compute gradient through computational graph"),
    ],
    "Section 5: Gradient Context Management": [
        ("test_no_grad_result", "Compute operation within torch.no_grad() context"),
        ("test_detached_result", "Detach tensor from computational graph"),
    ],
    "Section 6: Higher-Order Derivatives": [
        ("test_second_derivative", "Compute second derivative of x^4"),
    ],
}


def run_tests():
    """Run all tests for Exercise 1"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()