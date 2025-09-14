import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner

class Exercise1Validator(TestValidator):
    """Validator for Module 3 Exercise 1: Data Pipeline & Training Loop"""
    
    def test_simple_dataset(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if simple dataset is created correctly"""
        X_train = variables.get('X_train')
        y_train = variables.get('y_train')
        
        if X_train is None or y_train is None:
            return False, "X_train or y_train not found"
        
        if not isinstance(X_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
            return False, "X_train and y_train must be torch tensors"
        
        if X_train.shape[0] != 100 or X_train.shape[1] != 1:
            return False, f"Expected X_train shape (100, 1), got {X_train.shape}"
        
        if y_train.shape[0] != 100:
            return False, f"Expected y_train shape (100,) or (100, 1), got {y_train.shape}"
        
        return True, "Simple dataset created successfully"
    
    def test_dataset_splits(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if dataset is split into train/val/test correctly"""
        X_train_split = variables.get('X_train_split')
        X_val = variables.get('X_val')
        X_test = variables.get('X_test')
        
        if X_train_split is None or X_val is None or X_test is None:
            return False, "Dataset splits not found"
        
        total = len(X_train_split) + len(X_val) + len(X_test)
        if total != 100:
            return False, f"Total samples should be 100, got {total}"
        
        if len(X_train_split) != 60:
            return False, f"Expected 60 training samples, got {len(X_train_split)}"
        
        if len(X_val) != 20:
            return False, f"Expected 20 validation samples, got {len(X_val)}"
        
        if len(X_test) != 20:
            return False, f"Expected 20 test samples, got {len(X_test)}"
        
        return True, "Dataset splits created correctly"
    
    def test_tensor_dataset(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if TensorDataset is created correctly"""
        train_dataset = variables.get('train_dataset')
        
        if train_dataset is None:
            return False, "train_dataset not found"
        
        if not isinstance(train_dataset, TensorDataset):
            return False, f"Expected TensorDataset, got {type(train_dataset)}"
        
        if len(train_dataset) != 60:
            return False, f"Expected 60 samples in train_dataset, got {len(train_dataset)}"
        
        sample = train_dataset[0]
        if len(sample) != 2:
            return False, "Dataset should return (input, target) tuples"
        
        return True, "TensorDataset created successfully"
    
    def test_dataloader_batch8(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test DataLoader with batch size 8"""
        train_loader_8 = variables.get('train_loader_8')
        
        if train_loader_8 is None:
            return False, "train_loader_8 not found"
        
        if not isinstance(train_loader_8, DataLoader):
            return False, f"Expected DataLoader, got {type(train_loader_8)}"
        
        if train_loader_8.batch_size != 8:
            return False, f"Expected batch size 8, got {train_loader_8.batch_size}"
        
        first_batch = next(iter(train_loader_8))
        if first_batch[0].shape[0] != 8:
            return False, f"First batch should have 8 samples, got {first_batch[0].shape[0]}"
        
        return True, "DataLoader with batch size 8 created successfully"
    
    def test_dataloader_batch16(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test DataLoader with batch size 16"""
        train_loader_16 = variables.get('train_loader_16')
        
        if train_loader_16 is None:
            return False, "train_loader_16 not found"
        
        if not isinstance(train_loader_16, DataLoader):
            return False, f"Expected DataLoader, got {type(train_loader_16)}"
        
        if train_loader_16.batch_size != 16:
            return False, f"Expected batch size 16, got {train_loader_16.batch_size}"
        
        return True, "DataLoader with batch size 16 created successfully"
    
    def test_simple_model(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if simple linear model is created correctly"""
        model = variables.get('model')
        
        if model is None:
            return False, "model not found"
        
        if not isinstance(model, nn.Module):
            return False, f"Expected nn.Module, got {type(model)}"
        
        # Test forward pass
        test_input = torch.randn(1, 1)
        try:
            output = model(test_input)
            if output.shape != (1, 1):
                return False, f"Expected output shape (1, 1), got {output.shape}"
        except Exception as e:
            return False, f"Model forward pass failed: {e}"
        
        return True, "Simple model created successfully"
    
    def test_mse_loss(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if MSE loss is defined correctly"""
        loss_fn = variables.get('loss_fn')
        
        if loss_fn is None:
            return False, "loss_fn not found"
        
        if not isinstance(loss_fn, nn.MSELoss):
            return False, f"Expected nn.MSELoss, got {type(loss_fn)}"
        
        # Test loss computation
        pred = torch.tensor([[1.0]])
        target = torch.tensor([[2.0]])
        loss = loss_fn(pred, target)
        
        if not torch.allclose(loss, torch.tensor(1.0)):
            return False, f"MSE loss calculation incorrect: expected 1.0, got {loss.item()}"
        
        return True, "MSE loss defined correctly"
    
    def test_sgd_optimizer(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test SGD optimizer with learning rate 0.01"""
        optimizer_sgd = variables.get('optimizer_sgd')
        
        if optimizer_sgd is None:
            return False, "optimizer_sgd not found"
        
        if not isinstance(optimizer_sgd, torch.optim.SGD):
            return False, f"Expected torch.optim.SGD, got {type(optimizer_sgd)}"
        
        # Check learning rate
        for param_group in optimizer_sgd.param_groups:
            if param_group['lr'] != 0.01:
                return False, f"Expected learning rate 0.01, got {param_group['lr']}"
        
        return True, "SGD optimizer created correctly"
    
    def test_adam_optimizer(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test Adam optimizer with learning rate 0.001"""
        optimizer_adam = variables.get('optimizer_adam')
        
        if optimizer_adam is None:
            return False, "optimizer_adam not found"
        
        if not isinstance(optimizer_adam, torch.optim.Adam):
            return False, f"Expected torch.optim.Adam, got {type(optimizer_adam)}"
        
        # Check learning rate
        for param_group in optimizer_adam.param_groups:
            if param_group['lr'] != 0.001:
                return False, f"Expected learning rate 0.001, got {param_group['lr']}"
        
        return True, "Adam optimizer created correctly"
    
    def test_training_loop(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if training loop is implemented correctly"""
        train_losses = variables.get('train_losses')
        
        if train_losses is None:
            return False, "train_losses not found"
        
        if not isinstance(train_losses, list):
            return False, f"Expected list of losses, got {type(train_losses)}"
        
        if len(train_losses) != 10:
            return False, f"Expected 10 epochs of losses, got {len(train_losses)}"
        
        # Check if losses are decreasing overall
        if train_losses[-1] >= train_losses[0]:
            return False, "Training losses should decrease over epochs"
        
        return True, "Training loop implemented correctly"
    
    def test_evaluation_function(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if evaluation function is implemented"""
        evaluate_model = variables.get('evaluate_model')
        
        if evaluate_model is None:
            return False, "evaluate_model function not found"
        
        if not callable(evaluate_model):
            return False, "evaluate_model should be a function"
        
        # Test function signature by checking it can be called
        try:
            # Create dummy inputs
            model = nn.Linear(1, 1)
            loader = DataLoader(TensorDataset(torch.randn(10, 1), torch.randn(10, 1)), batch_size=2)
            loss_fn = nn.MSELoss()
            
            result = evaluate_model(model, loader, loss_fn)
            
            if not isinstance(result, (float, torch.Tensor)):
                return False, f"evaluate_model should return a float or tensor, got {type(result)}"
        except Exception as e:
            return False, f"evaluate_model function failed: {e}"
        
        return True, "Evaluation function implemented correctly"
    
    def test_train_val_losses(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if train and validation losses are tracked"""
        final_train_losses = variables.get('final_train_losses')
        final_val_losses = variables.get('final_val_losses')
        
        if final_train_losses is None or final_val_losses is None:
            return False, "final_train_losses or final_val_losses not found"
        
        if len(final_train_losses) != 20 or len(final_val_losses) != 20:
            return False, f"Expected 20 epochs of losses, got {len(final_train_losses)} train and {len(final_val_losses)} val"
        
        # Check if losses are reasonable
        if final_train_losses[-1] >= final_train_losses[0]:
            return False, "Training losses should generally decrease"
        
        return True, "Train and validation losses tracked correctly"
    
    def test_test_evaluation(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if model is evaluated on test set"""
        test_loss = variables.get('test_loss')
        
        if test_loss is None:
            return False, "test_loss not found"
        
        if not isinstance(test_loss, (float, torch.Tensor)):
            return False, f"Expected float or tensor for test_loss, got {type(test_loss)}"
        
        if isinstance(test_loss, torch.Tensor):
            test_loss = test_loss.item()
        
        if test_loss < 0:
            return False, f"Test loss should be positive, got {test_loss}"
        
        return True, "Test evaluation completed successfully"
    
    def test_learning_rate_comparison(self, variables: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if different learning rates are compared"""
        lr_results = variables.get('lr_results')
        
        if lr_results is None:
            return False, "lr_results not found"
        
        if not isinstance(lr_results, dict):
            return False, f"Expected dict for lr_results, got {type(lr_results)}"
        
        expected_lrs = [0.0001, 0.001, 0.01, 0.1]
        for lr in expected_lrs:
            if lr not in lr_results:
                return False, f"Missing results for learning rate {lr}"
            
            if not isinstance(lr_results[lr], list):
                return False, f"Expected list of losses for lr {lr}"
        
        return True, "Learning rate comparison completed successfully"

EXERCISE1_SECTIONS = {
    "Section 1: Creating a Simple Dataset": [
        ("test_simple_dataset", "Create tensors X_train and y_train with 100 samples"),
        ("test_dataset_splits", "Split data into train (60%), validation (20%), and test (20%)"),
    ],
    "Section 2: DataLoader with Different Batch Sizes": [
        ("test_tensor_dataset", "Create TensorDataset from training data"),
        ("test_dataloader_batch8", "Create DataLoader with batch size 8"),
        ("test_dataloader_batch16", "Create DataLoader with batch size 16"),
    ],
    "Section 3: Model and Loss Function": [
        ("test_simple_model", "Create a simple linear model"),
        ("test_mse_loss", "Define MSE loss function"),
    ],
    "Section 4: Optimizers": [
        ("test_sgd_optimizer", "Create SGD optimizer with lr=0.01"),
        ("test_adam_optimizer", "Create Adam optimizer with lr=0.001"),
    ],
    "Section 5: Training Loop": [
        ("test_training_loop", "Implement basic training loop for 10 epochs"),
    ],
    "Section 6: Evaluation on Train/Val/Test": [
        ("test_evaluation_function", "Create evaluation function"),
        ("test_train_val_losses", "Track training and validation losses"),
        ("test_test_evaluation", "Evaluate model on test set"),
    ],
    "Section 7: Learning Rate Comparison": [
        ("test_learning_rate_comparison", "Compare different learning rates"),
    ],
}

if __name__ == "__main__":
    validator = Exercise1Validator()
    runner = NotebookTestRunner("module3", 1)
    
    # Test with mock variables
    mock_vars = {
        'X_train': torch.randn(100, 1),
        'y_train': torch.randn(100),
        'X_train_split': torch.randn(60, 1),
        'X_val': torch.randn(20, 1),
        'X_test': torch.randn(20, 1),
        'train_dataset': TensorDataset(torch.randn(60, 1), torch.randn(60)),
        'train_loader_8': DataLoader(TensorDataset(torch.randn(60, 1), torch.randn(60)), batch_size=8),
        'train_loader_16': DataLoader(TensorDataset(torch.randn(60, 1), torch.randn(60)), batch_size=16),
        'model': nn.Linear(1, 1),
        'loss_fn': nn.MSELoss(),
        'optimizer_sgd': torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.01),
        'optimizer_adam': torch.optim.Adam(nn.Linear(1, 1).parameters(), lr=0.001),
        'train_losses': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        'evaluate_model': lambda m, l, f: 0.5,
        'final_train_losses': list(np.linspace(1.0, 0.1, 20)),
        'final_val_losses': list(np.linspace(1.1, 0.2, 20)),
        'test_loss': 0.15,
        'lr_results': {
            0.0001: [1.0, 0.95, 0.9],
            0.001: [1.0, 0.8, 0.6],
            0.01: [1.0, 0.5, 0.2],
            0.1: [1.0, 0.3, 0.1],
        }
    }
    
    print("Testing Module 3 Exercise 1 Validator...")
    for section, tests in EXERCISE1_SECTIONS.items():
        print(f"\n{section}:")
        for test_name, description in tests:
            test_method = getattr(validator, test_name)
            passed, message = test_method(mock_vars)
            status = "✓" if passed else "✗"
            print(f"  {status} {description}: {message}")