import sys
import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, Any

sys.path.append('..')
from test_utils import TestValidator, NotebookTestRunner


class Exercise2Validator(TestValidator):
    """Validator for Module 3 Exercise 2: Essential Layers"""
    
    # Section 1: Dropout Layer Tests
    def test_dropout_layer(self, variables: Dict[str, Any]) -> None:
        """Test the creation of a Dropout layer"""
        self.check_variable('dropout_layer', variables)
        dropout_layer = variables['dropout_layer']
        assert isinstance(dropout_layer, nn.Dropout), "dropout_layer should be an nn.Dropout instance"
        assert dropout_layer.p == 0.5, "Dropout probability should be 0.5"
    
    def test_dropout_train_mode(self, variables: Dict[str, Any]) -> None:
        """Test dropout behavior in training mode"""
        self.check_variable('output_train', variables)
        output_train = variables['output_train']
        self.check_tensor_shape(output_train, (10, 5), "output_train")
        # Check that some values are zero (dropout applied)
        assert (output_train == 0).any(), "In training mode, dropout should zero out some values"
    
    def test_dropout_eval_mode(self, variables: Dict[str, Any]) -> None:
        """Test dropout behavior in evaluation mode"""
        self.check_variable('output_eval', variables)
        output_eval = variables['output_eval']
        self.check_tensor_shape(output_eval, (10, 5), "output_eval")
        # Check that no values are exactly zero (dropout not applied)
        assert (output_eval != 0).all(), "In eval mode, dropout should not zero out any values"
    
    def test_model_with_dropout(self, variables: Dict[str, Any]) -> None:
        """Test the model with dropout layers"""
        self.check_variable('model_with_dropout', variables)
        model = variables['model_with_dropout']
        assert isinstance(model, nn.Module), "model_with_dropout should be an nn.Module"
        
        # Check model has the expected layers
        modules = list(model.children())
        has_linear = any(isinstance(m, nn.Linear) for m in modules)
        has_dropout = any(isinstance(m, nn.Dropout) for m in modules)
        has_relu = any(isinstance(m, nn.ReLU) for m in modules)
        
        assert has_linear, "Model should contain Linear layers"
        assert has_dropout, "Model should contain Dropout layers"
        assert has_relu, "Model should contain ReLU activation"
    
    # Section 2: Embedding Layer Tests
    def test_embedding_layer(self, variables: Dict[str, Any]) -> None:
        """Test the creation of an Embedding layer"""
        self.check_variable('embedding_layer', variables)
        embedding_layer = variables['embedding_layer']
        assert isinstance(embedding_layer, nn.Embedding), "embedding_layer should be an nn.Embedding instance"
        assert embedding_layer.num_embeddings == 100, "Vocabulary size should be 100"
        assert embedding_layer.embedding_dim == 16, "Embedding dimension should be 16"
    
    def test_word_indices(self, variables: Dict[str, Any]) -> None:
        """Test the word indices tensor"""
        self.check_variable('word_indices', variables)
        word_indices = variables['word_indices']
        self.check_tensor_shape(word_indices, (3, 5), "word_indices")
        self.check_tensor_dtype(word_indices, torch.long, "word_indices")
        # Check values are within vocabulary range
        assert word_indices.min() >= 0, "Word indices should be non-negative"
        assert word_indices.max() < 100, "Word indices should be less than vocabulary size (100)"
    
    def test_embedded_words(self, variables: Dict[str, Any]) -> None:
        """Test the embedded word vectors"""
        self.check_variable('embedded_words', variables)
        embedded_words = variables['embedded_words']
        self.check_tensor_shape(embedded_words, (3, 5, 16), "embedded_words")
        self.check_tensor_dtype(embedded_words, torch.float32, "embedded_words")
    
    def test_text_classifier(self, variables: Dict[str, Any]) -> None:
        """Test the text classifier with embeddings"""
        self.check_variable('text_classifier', variables)
        model = variables['text_classifier']
        assert isinstance(model, nn.Module), "text_classifier should be an nn.Module"
        
        # Test forward pass
        test_input = torch.randint(0, 1000, (2, 10), dtype=torch.long)
        output = model(test_input)
        assert output.shape == (2, 3), f"Expected output shape (2, 3), got {output.shape}"
    
    # Section 3: Skip Connections (Residual) Tests
    def test_residual_block(self, variables: Dict[str, Any]) -> None:
        """Test the ResidualBlock implementation"""
        self.check_variable('ResidualBlock', variables)
        ResidualBlock = variables['ResidualBlock']
        
        # Create an instance and test it
        block = ResidualBlock(64)
        assert isinstance(block, nn.Module), "ResidualBlock should be an nn.Module"
        
        # Test forward pass
        test_input = torch.randn(2, 64)
        output = block(test_input)
        assert output.shape == test_input.shape, "ResidualBlock should preserve input shape"
    
    def test_residual_output(self, variables: Dict[str, Any]) -> None:
        """Test the residual block output"""
        self.check_variable('residual_output', variables)
        residual_output = variables['residual_output']
        self.check_tensor_shape(residual_output, (4, 64), "residual_output")
    
    def test_deep_residual_net(self, variables: Dict[str, Any]) -> None:
        """Test the deep residual network"""
        self.check_variable('deep_residual_net', variables)
        model = variables['deep_residual_net']
        assert isinstance(model, nn.Module), "deep_residual_net should be an nn.Module"
        
        # Test forward pass
        test_input = torch.randn(2, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"
    
    # Section 4: Complete Model with Essential Layers Tests
    def test_complete_model(self, variables: Dict[str, Any]) -> None:
        """Test the complete model with all essential layers"""
        self.check_variable('complete_model', variables)
        model = variables['complete_model']
        assert isinstance(model, nn.Module), "complete_model should be an nn.Module"
        
        # Check that model uses various layer types
        modules_str = str(model)
        assert 'Embedding' in modules_str, "Model should include Embedding layer"
        assert 'Dropout' in modules_str, "Model should include Dropout layer"
        assert 'Linear' in modules_str, "Model should include Linear layers"
    
    def test_train_function(self, variables: Dict[str, Any]) -> None:
        """Test the training function"""
        self.check_variable('train_model', variables)
        train_model = variables['train_model']
        assert callable(train_model), "train_model should be a callable function"
    
    def test_training_losses(self, variables: Dict[str, Any]) -> None:
        """Test that training losses are tracked"""
        self.check_variable('training_losses', variables)
        training_losses = variables['training_losses']
        assert isinstance(training_losses, list), "training_losses should be a list"
        assert len(training_losses) > 0, "training_losses should not be empty"
        assert all(isinstance(loss, (float, torch.Tensor)) for loss in training_losses), \
            "All losses should be numeric values"
        
        # Check that losses generally decrease
        if len(training_losses) > 5:
            early_avg = np.mean(training_losses[:5])
            late_avg = np.mean(training_losses[-5:])
            assert late_avg <= early_avg * 1.1, "Training losses should generally decrease"
    
    def test_model_evaluation(self, variables: Dict[str, Any]) -> None:
        """Test model evaluation results"""
        self.check_variable('test_accuracy', variables)
        test_accuracy = variables['test_accuracy']
        assert isinstance(test_accuracy, (float, torch.Tensor)), "test_accuracy should be a numeric value"
        assert 0 <= float(test_accuracy) <= 1, "Accuracy should be between 0 and 1"


# Define test sections
EXERCISE2_SECTIONS = {
    "Section 1: Dropout Layer": [
        ("test_dropout_layer", "Create a Dropout layer with p=0.5"),
        ("test_dropout_train_mode", "Test dropout in training mode"),
        ("test_dropout_eval_mode", "Test dropout in evaluation mode"),
        ("test_model_with_dropout", "Create a model with dropout layers"),
    ],
    "Section 2: Embedding Layer": [
        ("test_embedding_layer", "Create an Embedding layer"),
        ("test_word_indices", "Create word indices tensor"),
        ("test_embedded_words", "Generate embedded word vectors"),
        ("test_text_classifier", "Create text classifier with embeddings"),
    ],
    "Section 3: Skip Connections": [
        ("test_residual_block", "Implement ResidualBlock class"),
        ("test_residual_output", "Test residual block forward pass"),
        ("test_deep_residual_net", "Create deep residual network"),
    ],
    "Section 4: Complete Model": [
        ("test_complete_model", "Create model with all essential layers"),
        ("test_train_function", "Implement training function"),
        ("test_training_losses", "Track training losses"),
        ("test_model_evaluation", "Evaluate model performance"),
    ],
}


if __name__ == "__main__":
    # Allow running specific test sections
    import sys
    if len(sys.argv) > 1:
        section = sys.argv[1]
        if section in EXERCISE2_SECTIONS:
            print(f"Testing {section}")
            validator = Exercise2Validator()
            # Would need actual variables to test
        else:
            print(f"Section '{section}' not found")
            print(f"Available sections: {list(EXERCISE2_SECTIONS.keys())}")
    else:
        print("Module 3 - Exercise 2: Essential Layers Test Suite")
        print(f"Sections: {list(EXERCISE2_SECTIONS.keys())}")