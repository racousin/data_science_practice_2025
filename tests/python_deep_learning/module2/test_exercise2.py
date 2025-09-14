import sys
import torch
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple, Callable

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner

class Exercise2Validator(TestValidator):
    """Validator for Module 2 Exercise 2: Optimization with PyTorch Autograd"""
    
    def test_f1_tensor(self, f1_tensor: Callable) -> Tuple[bool, str]:
        """Test the tensor version of function 1"""
        if not self.check_variable(f1_tensor, "f1_tensor"):
            return False, "f1_tensor function not defined"
        
        try:
            x = torch.tensor(0.0, requires_grad=True)
            result = f1_tensor(x)
            if not isinstance(result, torch.Tensor):
                return False, "f1_tensor should return a torch.Tensor"
            
            # Check values
            x_test = torch.tensor(0.0, requires_grad=True)
            if abs(f1_tensor(x_test).item() - 1.0) > 1e-6:
                return False, f"f1_tensor(0) should be 1.0, got {f1_tensor(x_test).item()}"
            
            x_test = torch.tensor(-1.0, requires_grad=True)
            if abs(f1_tensor(x_test).item() - 0.0) > 1e-6:
                return False, f"f1_tensor(-1) should be 0.0, got {f1_tensor(x_test).item()}"
            
            return True, "f1_tensor implemented correctly"
        except Exception as e:
            return False, f"Error testing f1_tensor: {str(e)}"
    
    def test_f1_gradient(self, f1_tensor: Callable) -> Tuple[bool, str]:
        """Test gradient computation for function 1"""
        if not self.check_variable(f1_tensor, "f1_tensor"):
            return False, "f1_tensor function not defined"
        
        try:
            # Test gradient at x=0
            x = torch.tensor(0.0, requires_grad=True)
            y = f1_tensor(x)
            y.backward()
            if abs(x.grad.item() - 2.0) > 1e-6:
                return False, f"Gradient at x=0 should be 2.0, got {x.grad.item()}"
            
            # Test gradient at x=-1 (minimum)
            x = torch.tensor(-1.0, requires_grad=True)
            y = f1_tensor(x)
            y.backward()
            if abs(x.grad.item()) > 1e-6:
                return False, f"Gradient at x=-1 should be 0.0, got {x.grad.item()}"
            
            return True, "f1_tensor gradient computation correct"
        except Exception as e:
            return False, f"Error testing f1_tensor gradient: {str(e)}"
    
    def test_f2_tensor(self, f2_tensor: Callable) -> Tuple[bool, str]:
        """Test the tensor version of function 2"""
        if not self.check_variable(f2_tensor, "f2_tensor"):
            return False, "f2_tensor function not defined"
        
        try:
            x = torch.tensor(0.0, requires_grad=True)
            result = f2_tensor(x)
            if not isinstance(result, torch.Tensor):
                return False, "f2_tensor should return a torch.Tensor"
            
            # Check values
            x_test = torch.tensor(0.0, requires_grad=True)
            if abs(f2_tensor(x_test).item() - 2.0) > 1e-6:
                return False, f"f2_tensor(0) should be 2.0, got {f2_tensor(x_test).item()}"
            
            x_test = torch.tensor(2.0, requires_grad=True)
            if abs(f2_tensor(x_test).item() + 2.0) > 1e-6:
                return False, f"f2_tensor(2) should be -2.0, got {f2_tensor(x_test).item()}"
            
            return True, "f2_tensor implemented correctly"
        except Exception as e:
            return False, f"Error testing f2_tensor: {str(e)}"
    
    def test_f2_gradient(self, f2_tensor: Callable) -> Tuple[bool, str]:
        """Test gradient computation for function 2"""
        if not self.check_variable(f2_tensor, "f2_tensor"):
            return False, "f2_tensor function not defined"
        
        try:
            # Test gradient at x=0
            x = torch.tensor(0.0, requires_grad=True)
            y = f2_tensor(x)
            y.backward()
            if abs(x.grad.item()) > 1e-6:
                return False, f"Gradient at x=0 should be 0.0, got {x.grad.item()}"
            
            # Test gradient at x=1
            x = torch.tensor(1.0, requires_grad=True)
            y = f2_tensor(x)
            y.backward()
            if abs(x.grad.item() + 3.0) > 1e-6:
                return False, f"Gradient at x=1 should be -3.0, got {x.grad.item()}"
            
            return True, "f2_tensor gradient computation correct"
        except Exception as e:
            return False, f"Error testing f2_tensor gradient: {str(e)}"
    
    def test_f3_tensor(self, f3_tensor: Callable) -> Tuple[bool, str]:
        """Test the tensor version of function 3"""
        if not self.check_variable(f3_tensor, "f3_tensor"):
            return False, "f3_tensor function not defined"
        
        try:
            point = torch.tensor([0.0, 0.0], requires_grad=True)
            result = f3_tensor(point)
            if not isinstance(result, torch.Tensor):
                return False, "f3_tensor should return a torch.Tensor"
            
            # Check values
            point_test = torch.tensor([0.0, 0.0], requires_grad=True)
            if abs(f3_tensor(point_test).item() - 5.0) > 1e-6:
                return False, f"f3_tensor([0, 0]) should be 5.0, got {f3_tensor(point_test).item()}"
            
            point_test = torch.tensor([-1.0, 1.0], requires_grad=True)
            if abs(f3_tensor(point_test).item() - 2.0) > 1e-6:
                return False, f"f3_tensor([-1, 1]) should be 2.0, got {f3_tensor(point_test).item()}"
            
            return True, "f3_tensor implemented correctly"
        except Exception as e:
            return False, f"Error testing f3_tensor: {str(e)}"
    
    def test_f3_gradient(self, f3_tensor: Callable) -> Tuple[bool, str]:
        """Test gradient computation for function 3"""
        if not self.check_variable(f3_tensor, "f3_tensor"):
            return False, "f3_tensor function not defined"
        
        try:
            # Test gradient at [0, 0]
            point = torch.tensor([0.0, 0.0], requires_grad=True)
            y = f3_tensor(point)
            y.backward()
            expected_grad = torch.tensor([2.0, -4.0])
            if torch.norm(point.grad - expected_grad) > 1e-6:
                return False, f"Gradient at [0, 0] should be [2, -4], got {point.grad.tolist()}"
            
            # Test gradient at minimum [-1, 1]
            point = torch.tensor([-1.0, 1.0], requires_grad=True)
            y = f3_tensor(point)
            y.backward()
            if torch.norm(point.grad) > 1e-6:
                return False, f"Gradient at [-1, 1] should be [0, 0], got {point.grad.tolist()}"
            
            return True, "f3_tensor gradient computation correct"
        except Exception as e:
            return False, f"Error testing f3_tensor gradient: {str(e)}"
    
    def test_gradient_descent_torch_1d(self, gradient_descent_torch_1d: Callable) -> Tuple[bool, str]:
        """Test gradient descent implementation for 1D functions"""
        if not self.check_variable(gradient_descent_torch_1d, "gradient_descent_torch_1d"):
            return False, "gradient_descent_torch_1d function not defined"
        
        try:
            # Simple quadratic function for testing
            def test_func(x):
                return x**2 + 2*x + 1
            
            result, trajectory = gradient_descent_torch_1d(
                test_func, x0=2.0, learning_rate=0.1, n_iterations=50
            )
            
            if result is None or trajectory is None:
                return False, "gradient_descent_torch_1d returned None"
            
            # Check convergence to minimum at x=-1
            if abs(result + 1.0) > 0.1:
                return False, f"Should converge near -1.0, got {result}"
            
            # Check trajectory structure
            if len(trajectory) != 51:  # Initial + 50 iterations
                return False, f"Trajectory should have 51 points, got {len(trajectory)}"
            
            return True, "gradient_descent_torch_1d implemented correctly"
        except Exception as e:
            return False, f"Error testing gradient_descent_torch_1d: {str(e)}"
    
    def test_gradient_descent_torch_nd(self, gradient_descent_torch_nd: Callable) -> Tuple[bool, str]:
        """Test gradient descent implementation for n-dimensional functions"""
        if not self.check_variable(gradient_descent_torch_nd, "gradient_descent_torch_nd"):
            return False, "gradient_descent_torch_nd function not defined"
        
        try:
            # Simple 2D quadratic function for testing
            def test_func(point):
                x, y = point[0], point[1]
                return x**2 + 2*y**2 + 2*x - 4*y + 5
            
            result, trajectory = gradient_descent_torch_nd(
                test_func, x0=torch.tensor([2.0, 2.0]), 
                learning_rate=0.1, n_iterations=50
            )
            
            if result is None or trajectory is None:
                return False, "gradient_descent_torch_nd returned None"
            
            # Check convergence to minimum at [-1, 1]
            expected = torch.tensor([-1.0, 1.0])
            if torch.norm(result - expected) > 0.1:
                return False, f"Should converge near [-1, 1], got {result.tolist()}"
            
            # Check trajectory structure
            if len(trajectory) != 51:  # Initial + 50 iterations
                return False, f"Trajectory should have 51 points, got {len(trajectory)}"
            
            return True, "gradient_descent_torch_nd implemented correctly"
        except Exception as e:
            return False, f"Error testing gradient_descent_torch_nd: {str(e)}"
    
    def test_sgd_optimizer(self, sgd_optimizer: Any, sgd_params: Any) -> Tuple[bool, str]:
        """Test SGD optimizer setup"""
        if not self.check_variable(sgd_optimizer, "sgd_optimizer"):
            return False, "sgd_optimizer not defined"
        if not self.check_variable(sgd_params, "sgd_params"):
            return False, "sgd_params not defined"
        
        try:
            # Check if it's a torch optimizer
            if not isinstance(sgd_optimizer, torch.optim.Optimizer):
                return False, "sgd_optimizer should be a torch.optim.Optimizer"
            
            # Check if it's specifically SGD
            if not isinstance(sgd_optimizer, torch.optim.SGD):
                return False, "sgd_optimizer should be torch.optim.SGD"
            
            # Check parameters
            if not isinstance(sgd_params, torch.Tensor):
                return False, "sgd_params should be a torch.Tensor"
            
            if not sgd_params.requires_grad:
                return False, "sgd_params should have requires_grad=True"
            
            return True, "SGD optimizer setup correctly"
        except Exception as e:
            return False, f"Error testing SGD optimizer: {str(e)}"
    
    def test_adam_optimizer(self, adam_optimizer: Any, adam_params: Any) -> Tuple[bool, str]:
        """Test Adam optimizer setup"""
        if not self.check_variable(adam_optimizer, "adam_optimizer"):
            return False, "adam_optimizer not defined"
        if not self.check_variable(adam_params, "adam_params"):
            return False, "adam_params not defined"
        
        try:
            # Check if it's a torch optimizer
            if not isinstance(adam_optimizer, torch.optim.Optimizer):
                return False, "adam_optimizer should be a torch.optim.Optimizer"
            
            # Check if it's specifically Adam
            if not isinstance(adam_optimizer, torch.optim.Adam):
                return False, "adam_optimizer should be torch.optim.Adam"
            
            # Check parameters
            if not isinstance(adam_params, torch.Tensor):
                return False, "adam_params should be a torch.Tensor"
            
            if not adam_params.requires_grad:
                return False, "adam_params should have requires_grad=True"
            
            return True, "Adam optimizer setup correctly"
        except Exception as e:
            return False, f"Error testing Adam optimizer: {str(e)}"
    
    def test_optimize_with_pytorch(self, optimize_with_pytorch: Callable) -> Tuple[bool, str]:
        """Test PyTorch optimizer usage function"""
        if not self.check_variable(optimize_with_pytorch, "optimize_with_pytorch"):
            return False, "optimize_with_pytorch function not defined"
        
        try:
            # Test with a simple function
            def test_func(x):
                return x**2 + 2*x + 1
            
            x0 = torch.tensor(2.0, requires_grad=True)
            optimizer = torch.optim.SGD([x0], lr=0.1)
            
            result, trajectory = optimize_with_pytorch(
                test_func, x0, optimizer, n_iterations=50
            )
            
            if result is None or trajectory is None:
                return False, "optimize_with_pytorch returned None"
            
            # Check convergence
            if abs(result.item() + 1.0) > 0.1:
                return False, f"Should converge near -1.0, got {result.item()}"
            
            # Check trajectory
            if len(trajectory) != 51:
                return False, f"Trajectory should have 51 points, got {len(trajectory)}"
            
            return True, "optimize_with_pytorch implemented correctly"
        except Exception as e:
            return False, f"Error testing optimize_with_pytorch: {str(e)}"
    
    def test_momentum_comparison(self, momentum_results: Dict) -> Tuple[bool, str]:
        """Test momentum comparison results"""
        if not self.check_variable(momentum_results, "momentum_results"):
            return False, "momentum_results not defined"
        
        try:
            if not isinstance(momentum_results, dict):
                return False, "momentum_results should be a dictionary"
            
            required_keys = ["no_momentum", "with_momentum"]
            for key in required_keys:
                if key not in momentum_results:
                    return False, f"momentum_results missing key: {key}"
                
                if "final" not in momentum_results[key]:
                    return False, f"momentum_results['{key}'] missing 'final' value"
                
                if "iterations" not in momentum_results[key]:
                    return False, f"momentum_results['{key}'] missing 'iterations' count"
            
            # Check that momentum helps convergence
            no_momentum_iters = momentum_results["no_momentum"]["iterations"]
            with_momentum_iters = momentum_results["with_momentum"]["iterations"]
            
            if with_momentum_iters >= no_momentum_iters:
                return False, "Momentum should reduce number of iterations needed"
            
            return True, "Momentum comparison completed correctly"
        except Exception as e:
            return False, f"Error testing momentum_comparison: {str(e)}"
    
    def test_learning_rate_scheduler(self, scheduler: Any, scheduler_results: List) -> Tuple[bool, str]:
        """Test learning rate scheduler implementation"""
        if not self.check_variable(scheduler, "scheduler"):
            return False, "scheduler not defined"
        if not self.check_variable(scheduler_results, "scheduler_results"):
            return False, "scheduler_results not defined"
        
        try:
            # Check if it's a learning rate scheduler
            if not hasattr(scheduler, 'step'):
                return False, "scheduler should have a 'step' method"
            
            # Check results structure
            if not isinstance(scheduler_results, list):
                return False, "scheduler_results should be a list"
            
            if len(scheduler_results) < 10:
                return False, "scheduler_results should have at least 10 learning rates recorded"
            
            # Check that learning rates decrease
            first_lr = scheduler_results[0]
            last_lr = scheduler_results[-1]
            
            if last_lr >= first_lr:
                return False, "Learning rate should decrease over time with scheduler"
            
            return True, "Learning rate scheduler implemented correctly"
        except Exception as e:
            return False, f"Error testing learning_rate_scheduler: {str(e)}"

# Define test sections
EXERCISE2_SECTIONS = {
    "Section 1: Function Definitions with Autograd": [
        ("test_f1_tensor", "Function 1 tensor implementation"),
        ("test_f1_gradient", "Function 1 gradient computation"),
        ("test_f2_tensor", "Function 2 tensor implementation"),
        ("test_f2_gradient", "Function 2 gradient computation"),
        ("test_f3_tensor", "Function 3 tensor implementation"),
        ("test_f3_gradient", "Function 3 gradient computation"),
    ],
    "Section 2: Gradient Descent with Autograd": [
        ("test_gradient_descent_torch_1d", "1D gradient descent with autograd"),
        ("test_gradient_descent_torch_nd", "N-D gradient descent with autograd"),
    ],
    "Section 3: PyTorch Optimizers": [
        ("test_sgd_optimizer", "SGD optimizer setup"),
        ("test_adam_optimizer", "Adam optimizer setup"),
        ("test_optimize_with_pytorch", "Optimization with PyTorch optimizers"),
    ],
    "Section 4: Advanced Optimization": [
        ("test_momentum_comparison", "Momentum comparison results"),
        ("test_learning_rate_scheduler", "Learning rate scheduler"),
    ],
}

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])