import sys
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple, Callable

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner


class Exercise2Validator(TestValidator):
    """Validator for Module 1 Exercise 2: Gradient Descent"""
    
    # Section 1: Function Definitions
    def test_f1(self) -> None:
        """Test function 1 implementation"""
        f1 = self.check_variable("f1", Callable)
        
        # Test function values
        assert abs(f1(0) - 1.0) < 1e-6, "f1(0) should equal 1"
        assert abs(f1(-1) - 0.0) < 1e-6, "f1(-1) should equal 0 (minimum)"
        assert abs(f1(1) - 4.0) < 1e-6, "f1(1) should equal 4"
        
    def test_grad_f1(self) -> None:
        """Test gradient of function 1"""
        grad_f1 = self.check_variable("grad_f1", Callable)
        
        # Test gradient values
        assert abs(grad_f1(0) - 2.0) < 1e-6, "grad_f1(0) should equal 2"
        assert abs(grad_f1(-1) - 0.0) < 1e-6, "grad_f1(-1) should equal 0 (at minimum)"
        assert abs(grad_f1(1) - 4.0) < 1e-6, "grad_f1(1) should equal 4"
    
    def test_f2(self) -> None:
        """Test function 2 implementation"""
        f2 = self.check_variable("f2", Callable)
        
        # Test function values
        assert abs(f2(0) - 2.0) < 1e-6, "f2(0) should equal 2"
        assert abs(f2(1) - 0.0) < 1e-6, "f2(1) should equal 0"
        assert abs(f2(2) - (-2.0)) < 1e-6, "f2(2) should equal -2 (local minimum)"
        
    def test_grad_f2(self) -> None:
        """Test gradient of function 2"""
        grad_f2 = self.check_variable("grad_f2", Callable)
        
        # Test gradient values
        assert abs(grad_f2(0) - 0.0) < 1e-6, "grad_f2(0) should equal 0 (critical point)"
        assert abs(grad_f2(2) - 0.0) < 1e-6, "grad_f2(2) should equal 0 (at local minimum)"
        assert abs(grad_f2(1) - (-3.0)) < 1e-6, "grad_f2(1) should equal -3"
    
    def test_f3(self) -> None:
        """Test function 3 implementation"""
        f3 = self.check_variable("f3", Callable)
        
        # Test function values
        point1 = np.array([0, 0])
        assert abs(f3(point1) - 5.0) < 1e-6, "f3([0,0]) should equal 5"
        
        point2 = np.array([-1, 1])
        assert abs(f3(point2) - 2.0) < 1e-6, "f3([-1,1]) should equal 2 (minimum)"
        
        point3 = np.array([1, 1])
        assert abs(f3(point3) - 6.0) < 1e-6, "f3([1,1]) should equal 6"
        
    def test_grad_f3(self) -> None:
        """Test gradient of function 3"""
        grad_f3 = self.check_variable("grad_f3", Callable)
        
        # Test gradient values
        point1 = np.array([0, 0])
        grad1 = grad_f3(point1)
        assert isinstance(grad1, np.ndarray), "grad_f3 should return numpy array"
        assert grad1.shape == (2,), "grad_f3 should return array of shape (2,)"
        assert abs(grad1[0] - 2.0) < 1e-6, "grad_f3([0,0])[0] should equal 2"
        assert abs(grad1[1] - (-4.0)) < 1e-6, "grad_f3([0,0])[1] should equal -4"
        
        point2 = np.array([-1, 1])
        grad2 = grad_f3(point2)
        assert abs(grad2[0] - 0.0) < 1e-6, "grad_f3([-1,1])[0] should equal 0 (at minimum)"
        assert abs(grad2[1] - 0.0) < 1e-6, "grad_f3([-1,1])[1] should equal 0 (at minimum)"
    
    # Section 3: Gradient Descent Implementation
    def test_gradient_descent_1d(self) -> None:
        """Test 1D gradient descent implementation"""
        gradient_descent_1d = self.check_variable("gradient_descent_1d", Callable)
        
        # Simple quadratic test: f(x) = (x-1)^2, gradient = 2(x-1)
        def test_f(x): return (x - 1) ** 2
        def test_grad(x): return 2 * (x - 1)
        
        final_x, trajectory = gradient_descent_1d(test_f, test_grad, x0=5.0, 
                                                   learning_rate=0.1, n_iterations=50)
        
        assert final_x is not None, "gradient_descent_1d should return final position"
        assert trajectory is not None, "gradient_descent_1d should return trajectory"
        assert len(trajectory) == 51, "Trajectory should include initial position plus all iterations"
        assert abs(trajectory[0] - 5.0) < 1e-6, "First trajectory point should be x0"
        assert abs(final_x - 1.0) < 0.01, f"Should converge near minimum at x=1, got {final_x}"
        
        # Check that trajectory is descending
        for i in range(1, len(trajectory)):
            assert trajectory[i] <= trajectory[i-1] + 1e-10, "Trajectory should be descending"
    
    def test_gradient_descent_nd(self) -> None:
        """Test n-dimensional gradient descent implementation"""
        gradient_descent_nd = self.check_variable("gradient_descent_nd", Callable)
        
        # Simple 2D quadratic: f(x,y) = (x-1)^2 + (y-2)^2
        def test_f(point): 
            x, y = point
            return (x - 1) ** 2 + (y - 2) ** 2
        
        def test_grad(point):
            x, y = point
            return np.array([2 * (x - 1), 2 * (y - 2)])
        
        x0 = np.array([5.0, 6.0])
        final_x, trajectory = gradient_descent_nd(test_f, test_grad, x0=x0,
                                                  learning_rate=0.1, n_iterations=50)
        
        assert final_x is not None, "gradient_descent_nd should return final position"
        assert trajectory is not None, "gradient_descent_nd should return trajectory"
        
        trajectory = np.array(trajectory)
        assert trajectory.shape[0] == 51, "Trajectory should include initial position plus all iterations"
        assert trajectory.shape[1] == 2, "Trajectory should be 2D"
        assert np.allclose(trajectory[0], x0), "First trajectory point should be x0"
        assert abs(final_x[0] - 1.0) < 0.01, f"Should converge near x=1, got {final_x[0]}"
        assert abs(final_x[1] - 2.0) < 0.01, f"Should converge near y=2, got {final_x[1]}"
    
    def test_result_1d(self) -> None:
        """Test gradient descent result on function 1"""
        result_1d = self.namespace.get("result_1d")
        trajectory_1d = self.namespace.get("trajectory_1d")
        
        if result_1d is None or trajectory_1d is None:
            pytest.skip("gradient_descent_1d not yet implemented")
            
        assert abs(result_1d - (-1.0)) < 0.1, f"Should converge near minimum at x=-1, got {result_1d}"
        assert len(trajectory_1d) > 0, "Trajectory should not be empty"
    
    def test_result_nd(self) -> None:
        """Test gradient descent result on function 3"""
        result_nd = self.namespace.get("result_nd")
        trajectory_nd = self.namespace.get("trajectory_nd")
        
        if result_nd is None or trajectory_nd is None:
            pytest.skip("gradient_descent_nd not yet implemented")
            
        assert abs(result_nd[0] - (-1.0)) < 0.1, f"Should converge near x=-1, got {result_nd[0]}"
        assert abs(result_nd[1] - 1.0) < 0.1, f"Should converge near y=1, got {result_nd[1]}"
    
    # Section 4: Learning Rate Experiments
    def test_optimal_lr_f1(self) -> None:
        """Test optimal learning rate for function 1"""
        optimal_lr_f1 = self.check_variable("optimal_lr_f1", float)
        
        # For quadratic functions, optimal is around 0.1-0.5
        assert 0.05 <= optimal_lr_f1 <= 0.8, f"Optimal LR for f1 should be between 0.05 and 0.8, got {optimal_lr_f1}"
    
    def test_optimal_lr_f2(self) -> None:
        """Test optimal learning rate for function 2"""
        optimal_lr_f2 = self.check_variable("optimal_lr_f2", float)
        
        # For cubic functions, need smaller learning rate
        assert 0.01 <= optimal_lr_f2 <= 0.3, f"Optimal LR for f2 should be between 0.01 and 0.3, got {optimal_lr_f2}"
    
    def test_optimal_lr_f3(self) -> None:
        """Test optimal learning rate for function 3"""
        optimal_lr_f3 = self.check_variable("optimal_lr_f3", float)
        
        # For 2D convex function
        assert 0.05 <= optimal_lr_f3 <= 0.4, f"Optimal LR for f3 should be between 0.05 and 0.4, got {optimal_lr_f3}"
    
    # Section 5: Convergence Analysis
    def test_check_convergence(self) -> None:
        """Test convergence checking function"""
        check_convergence = self.check_variable("check_convergence", Callable)
        
        # Test converged trajectory
        converged_traj = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 
                         0.0078125, 0.00390625, 0.001953125, 0.001, 0.001, 0.001]
        
        conv_iter = check_convergence(converged_traj, threshold=0.001)
        assert conv_iter == 10, f"Should detect convergence at iteration 10, got {conv_iter}"
        
        # Test non-converged trajectory
        non_converged = [1.0, 0.5, 0.25, 0.125, 0.0625]
        conv_iter = check_convergence(non_converged, threshold=0.001)
        assert conv_iter == -1, "Should return -1 for non-converged trajectory"
        
        # Test immediate convergence
        immediate = [1.0, 1.0, 1.0]
        conv_iter = check_convergence(immediate, threshold=0.001)
        assert conv_iter == 1, "Should detect immediate convergence"


# Define test sections
EXERCISE2_SECTIONS = {
    "Section 1: Function Definitions": [
        ("test_f1", "Testing function 1 implementation"),
        ("test_grad_f1", "Testing gradient of function 1"),
        ("test_f2", "Testing function 2 implementation"),
        ("test_grad_f2", "Testing gradient of function 2"),
        ("test_f3", "Testing function 3 implementation"),
        ("test_grad_f3", "Testing gradient of function 3"),
    ],
    "Section 3: Gradient Descent 1D": [
        ("test_gradient_descent_1d", "Testing 1D gradient descent implementation"),
        ("test_result_1d", "Testing convergence on function 1"),
    ],
    "Section 3: Gradient Descent ND": [
        ("test_gradient_descent_nd", "Testing n-dimensional gradient descent"),
        ("test_result_nd", "Testing convergence on function 3"),
    ],
    "Section 4: Learning Rate Experiments": [
        ("test_optimal_lr_f1", "Testing optimal learning rate for function 1"),
        ("test_optimal_lr_f2", "Testing optimal learning rate for function 2"),
        ("test_optimal_lr_f3", "Testing optimal learning rate for function 3"),
    ],
    "Section 5: Convergence Analysis": [
        ("test_check_convergence", "Testing convergence detection function"),
    ],
}


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])