#!/usr/bin/env python3
"""
Unified test utilities for Python Deep Learning course.
This module provides a flexible testing framework that works both in pytest and Colab notebooks.
"""

import sys
import torch
import numpy as np
from typing import Dict, Any, Callable, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TestStatus(Enum):
    """Test execution status"""
    PASSED = "✅"
    FAILED = "❌"
    SKIPPED = "⏭️"
    WARNING = "⚠️"


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    status: TestStatus
    message: str = ""
    details: Optional[str] = None


class TestValidator:
    """Base validator class for exercise tests"""
    
    def __init__(self, namespace: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with optional namespace (for notebook testing).
        
        Args:
            namespace: Dictionary containing variables to test (typically locals() in notebook)
        """
        self.namespace = namespace or {}
        self.results: List[TestResult] = []
        
    def check_variable(self, var_name: str, expected_type: Optional[type] = None) -> Any:
        """
        Check if a variable exists in namespace and optionally validate its type.
        
        Args:
            var_name: Name of the variable to check
            expected_type: Expected type of the variable (optional)
            
        Returns:
            The variable value if it exists and passes type check
            
        Raises:
            AssertionError: If variable doesn't exist or has wrong type
        """
        if var_name not in self.namespace:
            raise AssertionError(f"{var_name} not found. Please create it according to the instructions.")
        
        value = self.namespace[var_name]
        
        if expected_type is not None and not isinstance(value, expected_type):
            raise AssertionError(
                f"{var_name} should be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        
        return value
    
    def check_tensor_shape(self, tensor: torch.Tensor, expected_shape: Tuple, name: str = "tensor"):
        """Check if tensor has expected shape"""
        if tensor.shape != torch.Size(expected_shape):
            raise AssertionError(
                f"{name} should have shape {expected_shape}, got {tensor.shape}"
            )
    
    def check_tensor_dtype(self, tensor: torch.Tensor, expected_dtype: torch.dtype, name: str = "tensor"):
        """Check if tensor has expected data type"""
        if tensor.dtype != expected_dtype:
            raise AssertionError(
                f"{name} should have dtype {expected_dtype}, got {tensor.dtype}"
            )
    
    def check_tensor_values(self, tensor: torch.Tensor, expected: torch.Tensor, name: str = "tensor", 
                          tolerance: float = 1e-6):
        """Check if tensor values match expected (with tolerance for floating point)"""
        if tensor.dtype in [torch.float32, torch.float64]:
            if not torch.allclose(tensor, expected, atol=tolerance):
                raise AssertionError(f"{name} values don't match expected")
        else:
            if not torch.equal(tensor, expected):
                raise AssertionError(f"{name} values don't match expected")
    
    def run_test(self, test_func: Callable, test_name: str) -> TestResult:
        """
        Run a single test function and capture the result.
        
        Args:
            test_func: Test function to run
            test_name: Display name for the test
            
        Returns:
            TestResult object with status and details
        """
        try:
            test_func()
            return TestResult(name=test_name, status=TestStatus.PASSED)
        except AssertionError as e:
            return TestResult(
                name=test_name, 
                status=TestStatus.FAILED,
                message=str(e)
            )
        except Exception as e:
            return TestResult(
                name=test_name,
                status=TestStatus.FAILED,
                message=f"Unexpected error: {e}",
                details=str(type(e).__name__)
            )
    
    def print_results(self, verbose: bool = True) -> bool:
        """
        Print test results summary.
        
        Args:
            verbose: If True, show details for failed tests
            
        Returns:
            True if all tests passed, False otherwise
        """
        if not self.results:
            print("No tests have been run.")
            return True
        
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        for result in self.results:
            status_icon = result.status.value
            print(f"{status_icon} {result.name}")
            if verbose and result.status == TestStatus.FAILED:
                print(f"   └─ {result.message}")
                if result.details:
                    print(f"      └─ {result.details}")
        
        print("-"*60)
        print(f"Total: {len(self.results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
        
        if failed == 0:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"❌ {failed} test(s) failed. Please review and fix the issues above.")
            return False


class NotebookTestRunner:
    """
    Interactive test runner for Jupyter/Colab notebooks.
    Provides section-by-section testing with immediate feedback.
    """
    
    def __init__(self, module_name: str, exercise_num: int):
        """
        Initialize the notebook test runner.
        
        Args:
            module_name: Name of the module (e.g., "module1")
            exercise_num: Exercise number
        """
        self.module_name = module_name
        self.exercise_num = exercise_num
        self.section_results: Dict[str, List[TestResult]] = {}
        
    def test_section(self, section_name: str, validator: TestValidator, 
                     tests: List[Tuple[Callable, str]], 
                     namespace: Optional[Dict[str, Any]] = None) -> bool:
        """
        Run tests for a specific section.
        
        Args:
            section_name: Name of the section being tested
            validator: TestValidator instance to use
            tests: List of (test_function, test_name) tuples
            namespace: Optional namespace to update validator with
            
        Returns:
            True if all tests passed, False otherwise
        """
        if namespace:
            validator.namespace = namespace
        
        print(f"\n{'='*50}")
        print(f"Testing: {section_name}")
        print('='*50)
        
        results = []
        all_passed = True
        
        for test_func, test_name in tests:
            result = validator.run_test(test_func, test_name)
            results.append(result)
            
            # Print immediate feedback
            status_icon = result.status.value
            if result.status == TestStatus.PASSED:
                print(f"{status_icon} {test_name}")
            else:
                print(f"{status_icon} {test_name}: {result.message}")
                all_passed = False
        
        self.section_results[section_name] = results
        
        if all_passed:
            print(f"\n✅ {section_name} - All tests passed!")
        else:
            print(f"\n❌ {section_name} - Some tests failed. Review the errors above.")
        
        return all_passed
    
    def final_summary(self) -> bool:
        """
        Print final summary of all sections tested.
        
        Returns:
            True if all sections passed, False otherwise
        """
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        
        all_passed = True
        
        for section_name, results in self.section_results.items():
            passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            total = len(results)
            
            if passed == total:
                print(f"✅ {section_name}: {passed}/{total} tests passed")
            else:
                print(f"❌ {section_name}: {passed}/{total} tests passed")
                all_passed = False
        
        print("="*60)
        
        if all_passed:
            print(f"🎉 Congratulations! All tests passed for Exercise {self.exercise_num}!")
            print("You're ready to move on to the next exercise.")
        else:
            print("❌ Some tests are still failing. Please review and complete the TODOs.")
        
        return all_passed


def run_pytest_test(test_class, method_name: str = None):
    """
    Run pytest tests programmatically (useful for running specific test methods).
    
    Args:
        test_class: The test class to run
        method_name: Optional specific method to run
    """
    import pytest
    
    test_instance = test_class()
    if hasattr(test_instance, 'setup_method'):
        test_instance.setup_method()
    
    if method_name:
        method = getattr(test_instance, method_name)
        method()
        print(f"✅ {method_name} passed")
    else:
        # Run all test methods
        for attr_name in dir(test_instance):
            if attr_name.startswith('test_'):
                method = getattr(test_instance, attr_name)
                try:
                    method()
                    print(f"✅ {attr_name} passed")
                except Exception as e:
                    print(f"❌ {attr_name} failed: {e}")


def create_inline_test(description: str, test_func: Callable, namespace: Dict[str, Any]) -> bool:
    """
    Create and run an inline test with nice formatting for notebooks.
    
    Args:
        description: Description of what's being tested
        test_func: Function that performs the test
        namespace: Namespace containing variables to test
        
    Returns:
        True if test passed, False otherwise
    """
    try:
        test_func(namespace)
        print(f"✅ {description}")
        return True
    except AssertionError as e:
        print(f"❌ {description}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ {description}")
        print(f"   Unexpected error: {type(e).__name__}: {e}")
        return False