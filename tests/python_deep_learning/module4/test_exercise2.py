import sys
import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner


class Exercise2Validator(TestValidator):
    """Validator for Module 4 Exercise 2: Fine-Tuning"""
    
    def test_simple_pretrained_model(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test if simple pretrained model is created correctly"""
        model = self.check_variable(context, 'simple_pretrained_model')
        
        if not isinstance(model, nn.Module):
            return False, "simple_pretrained_model should be a torch.nn.Module"
        
        # Check if model has expected layers
        if not hasattr(model, 'features') or not hasattr(model, 'classifier'):
            return False, "Model should have 'features' and 'classifier' attributes"
        
        # Check if features are frozen
        features_frozen = all(not param.requires_grad for param in model.features.parameters())
        if not features_frozen:
            return False, "Feature layers should be frozen (requires_grad=False)"
        
        # Check if classifier is trainable
        classifier_trainable = all(param.requires_grad for param in model.classifier.parameters())
        if not classifier_trainable:
            return False, "Classifier layers should be trainable (requires_grad=True)"
        
        return True, "Simple pretrained model created correctly with frozen features"
    
    def test_feature_extractor(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test feature extraction setup"""
        extractor = self.check_variable(context, 'feature_extractor')
        
        if not isinstance(extractor, nn.Module):
            return False, "feature_extractor should be a torch.nn.Module"
        
        # Test that it's in eval mode
        if extractor.training:
            return False, "Feature extractor should be in eval mode"
        
        # Check that parameters don't require gradients
        all_frozen = all(not param.requires_grad for param in extractor.parameters())
        if not all_frozen:
            return False, "All feature extractor parameters should have requires_grad=False"
        
        return True, "Feature extractor configured correctly"
    
    def test_extracted_features(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test extracted features shape and properties"""
        features = self.check_variable(context, 'extracted_features')
        
        if not isinstance(features, torch.Tensor):
            return False, "extracted_features should be a torch.Tensor"
        
        if features.dim() != 2:
            return False, f"Features should be 2D (batch_size, feature_dim), got {features.dim()}D"
        
        if features.shape[0] != 32:  # Expected batch size
            return False, f"Batch size should be 32, got {features.shape[0]}"
        
        if features.shape[1] < 64:  # Minimum expected feature dimension
            return False, f"Feature dimension seems too small: {features.shape[1]}"
        
        return True, "Features extracted correctly"
    
    def test_fine_tuned_model(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test fine-tuned model structure"""
        model = self.check_variable(context, 'fine_tuned_model')
        
        if not isinstance(model, nn.Module):
            return False, "fine_tuned_model should be a torch.nn.Module"
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        if trainable_params == 0:
            return False, "Model should have trainable parameters"
        
        if trainable_params == total_params:
            return False, "Not all parameters should be trainable (some should be frozen)"
        
        return True, f"Fine-tuned model has {trainable_params}/{total_params} trainable parameters"
    
    def test_layer_freezing_strategy(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test selective layer freezing implementation"""
        freeze_layers = self.check_variable(context, 'freeze_layers')
        
        if not callable(freeze_layers):
            return False, "freeze_layers should be a callable function"
        
        # Test on a dummy model
        test_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 30),
            nn.Linear(30, 10)
        )
        
        # Apply freezing
        try:
            freeze_layers(test_model, num_layers_to_freeze=2)
        except Exception as e:
            return False, f"Error in freeze_layers function: {str(e)}"
        
        # Check if first 2 layers are frozen
        layers = list(test_model.children())
        for i in range(2):
            if any(p.requires_grad for p in layers[i].parameters()):
                return False, f"Layer {i} should be frozen but has trainable parameters"
        
        # Check if last layer is trainable
        if not any(p.requires_grad for p in layers[2].parameters()):
            return False, "Last layer should remain trainable"
        
        return True, "Layer freezing strategy implemented correctly"
    
    def test_gradual_unfreezing(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test gradual unfreezing implementation"""
        unfreeze_schedule = self.check_variable(context, 'unfreeze_schedule')
        
        if not isinstance(unfreeze_schedule, list):
            return False, "unfreeze_schedule should be a list"
        
        if len(unfreeze_schedule) < 3:
            return False, "unfreeze_schedule should have at least 3 epochs"
        
        # Check if schedule makes sense (epochs should be increasing)
        for i in range(1, len(unfreeze_schedule)):
            if unfreeze_schedule[i][0] <= unfreeze_schedule[i-1][0]:
                return False, "Epochs in unfreeze_schedule should be in increasing order"
        
        return True, "Gradual unfreezing schedule defined correctly"
    
    def test_learning_rate_schedule(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test discriminative learning rates"""
        lr_groups = self.check_variable(context, 'lr_groups')
        
        if not isinstance(lr_groups, list):
            return False, "lr_groups should be a list"
        
        if len(lr_groups) < 2:
            return False, "Should have at least 2 learning rate groups"
        
        # Check if learning rates are different
        lrs = [group['lr'] for group in lr_groups]
        if len(set(lrs)) == 1:
            return False, "Different parameter groups should have different learning rates"
        
        # Check if early layers have lower learning rates
        if lrs[0] >= lrs[-1]:
            return False, "Early layers should have lower learning rates than later layers"
        
        return True, "Discriminative learning rates configured correctly"
    
    def test_fine_tuning_loss(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test fine-tuning loss calculation"""
        initial_loss = self.check_variable(context, 'initial_loss')
        final_loss = self.check_variable(context, 'final_loss')
        
        if not isinstance(initial_loss, (float, torch.Tensor)):
            return False, "initial_loss should be a float or tensor"
        
        if not isinstance(final_loss, (float, torch.Tensor)):
            return False, "final_loss should be a float or tensor"
        
        initial_val = initial_loss.item() if isinstance(initial_loss, torch.Tensor) else initial_loss
        final_val = final_loss.item() if isinstance(final_loss, torch.Tensor) else final_loss
        
        if final_val >= initial_val:
            return False, f"Loss should decrease during fine-tuning. Initial: {initial_val:.4f}, Final: {final_val:.4f}"
        
        improvement = (initial_val - final_val) / initial_val * 100
        return True, f"Loss improved by {improvement:.1f}%"
    
    def test_adapter_module(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test adapter module implementation"""
        adapter = self.check_variable(context, 'AdapterModule')
        
        if not isinstance(adapter, type):
            return False, "AdapterModule should be a class"
        
        # Try to instantiate
        try:
            adapter_instance = adapter(512)  # Common feature dimension
        except Exception as e:
            return False, f"Could not instantiate AdapterModule: {str(e)}"
        
        if not isinstance(adapter_instance, nn.Module):
            return False, "AdapterModule should inherit from nn.Module"
        
        # Count parameters
        adapter_params = sum(p.numel() for p in adapter_instance.parameters())
        if adapter_params > 512 * 512:  # Should be small
            return False, f"Adapter has too many parameters ({adapter_params}). Should be lightweight"
        
        return True, f"Adapter module implemented with {adapter_params} parameters"
    
    def test_lora_implementation(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test LoRA (Low-Rank Adaptation) implementation"""
        lora_layer = self.check_variable(context, 'LoRALinear')
        
        if not isinstance(lora_layer, type):
            return False, "LoRALinear should be a class"
        
        # Try to instantiate
        try:
            lora_instance = lora_layer(256, 128, rank=8)
        except Exception as e:
            return False, f"Could not instantiate LoRALinear: {str(e)}"
        
        if not isinstance(lora_instance, nn.Module):
            return False, "LoRALinear should inherit from nn.Module"
        
        # Check for LoRA components
        if not hasattr(lora_instance, 'lora_A') or not hasattr(lora_instance, 'lora_B'):
            return False, "LoRALinear should have lora_A and lora_B components"
        
        # Check rank
        if lora_instance.lora_A.shape[1] != 8 or lora_instance.lora_B.shape[0] != 8:
            return False, "LoRA rank not correctly implemented"
        
        return True, "LoRA implementation correct with low-rank decomposition"
    
    def test_huggingface_model_loading(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test Hugging Face model loading setup"""
        model_name = self.check_variable(context, 'hf_model_name')
        
        if not isinstance(model_name, str):
            return False, "hf_model_name should be a string"
        
        if 'bert' not in model_name.lower() and 'distil' not in model_name.lower() and 'tiny' not in model_name.lower():
            return False, "Please use a small model like 'distilbert' or 'tiny-bert' for this exercise"
        
        return True, f"Hugging Face model name set to: {model_name}"
    
    def test_tokenizer_setup(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test tokenizer configuration"""
        max_length = self.check_variable(context, 'max_length')
        
        if not isinstance(max_length, int):
            return False, "max_length should be an integer"
        
        if max_length < 32 or max_length > 512:
            return False, f"max_length should be between 32 and 512, got {max_length}"
        
        return True, f"Tokenizer max_length set to {max_length}"
    
    def test_classification_head(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test custom classification head"""
        classification_head = self.check_variable(context, 'classification_head')
        
        if not isinstance(classification_head, nn.Module):
            return False, "classification_head should be a torch.nn.Module"
        
        # Check for dropout
        has_dropout = any(isinstance(m, nn.Dropout) for m in classification_head.modules())
        if not has_dropout:
            return False, "Classification head should include dropout for regularization"
        
        # Check output dimension
        try:
            test_input = torch.randn(1, 768)  # Common hidden size
            output = classification_head(test_input)
            if output.shape[1] < 2:
                return False, f"Classification head should output at least 2 classes, got {output.shape[1]}"
        except:
            pass  # May have different input size
        
        return True, "Classification head configured correctly"
    
    def test_training_config(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Test fine-tuning training configuration"""
        config = self.check_variable(context, 'fine_tuning_config')
        
        if not isinstance(config, dict):
            return False, "fine_tuning_config should be a dictionary"
        
        required_keys = ['learning_rate', 'batch_size', 'num_epochs', 'warmup_steps']
        missing_keys = [k for k in required_keys if k not in config]
        
        if missing_keys:
            return False, f"Missing configuration keys: {missing_keys}"
        
        # Check reasonable values
        if config['learning_rate'] > 1e-3:
            return False, "Learning rate too high for fine-tuning (should be < 1e-3)"
        
        if config['num_epochs'] > 10:
            return False, "Too many epochs for fine-tuning (should be <= 10)"
        
        return True, "Fine-tuning configuration set appropriately"


EXERCISE2_SECTIONS = {
    "Section 1: Feature Extraction Basics": [
        ("test_simple_pretrained_model", "Simple pretrained model with frozen features"),
        ("test_feature_extractor", "Feature extractor configuration"),
        ("test_extracted_features", "Extracted features shape and properties")
    ],
    "Section 2: Fine-Tuning Strategies": [
        ("test_fine_tuned_model", "Fine-tuned model structure"),
        ("test_layer_freezing_strategy", "Selective layer freezing"),
        ("test_gradual_unfreezing", "Gradual unfreezing schedule")
    ],
    "Section 3: Advanced Techniques": [
        ("test_learning_rate_schedule", "Discriminative learning rates"),
        ("test_fine_tuning_loss", "Fine-tuning loss improvement"),
        ("test_adapter_module", "Adapter module implementation"),
        ("test_lora_implementation", "LoRA (Low-Rank Adaptation)")
    ],
    "Section 4: Hugging Face Integration": [
        ("test_huggingface_model_loading", "Hugging Face model selection"),
        ("test_tokenizer_setup", "Tokenizer configuration"),
        ("test_classification_head", "Custom classification head"),
        ("test_training_config", "Fine-tuning training configuration")
    ]
}


if __name__ == "__main__":
    validator = Exercise2Validator()
    runner = NotebookTestRunner("module4", 2)
    
    # Test with sample context
    sample_context = {
        'simple_pretrained_model': nn.Sequential(
            nn.Sequential(nn.Linear(10, 20), nn.ReLU()),
            nn.Linear(20, 2)
        ),
        'max_length': 128,
        'hf_model_name': 'distilbert-base-uncased',
        'fine_tuning_config': {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'num_epochs': 3,
            'warmup_steps': 100
        }
    }
    
    print("Testing Module 4 Exercise 2: Fine-Tuning")
    print("=" * 50)
    
    for section_name, tests in EXERCISE2_SECTIONS.items():
        print(f"\n{section_name}")
        print("-" * 40)
        for test_name, description in tests:
            test_method = getattr(validator, test_name)
            success, message = test_method(sample_context)
            status = "✓" if success else "✗"
            print(f"  {status} {description}: {message}")