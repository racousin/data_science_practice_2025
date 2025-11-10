import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

import numpy as np

def weighted_accuracy(y_true, y_pred):
    weights = np.abs(y_true)

    # Compute the sign of true and predicted values
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)

    # Correct predictions where the sign of the true and predicted values match
    correct_predictions = sign_true == sign_pred

    # Compute the weighted accuracy
    weighted_acc = np.sum(weights * correct_predictions) / np.sum(weights)

    return weighted_acc

def rounded_accuracy(y_true, y_pred, decimals=2):
    """
    Compute accuracy after rounding both true values and predictions to specified decimal places.

    This metric computes the proportion of predictions that exactly match the true values
    after both are rounded to the specified number of decimal places.

    Args:
        y_true: True values
        y_pred: Predicted values
        decimals: Number of decimal places to round to (default: 2)

    Returns:
        Accuracy score (proportion of exact matches after rounding)
    """
    y_true_rounded = np.round(y_true, decimals=decimals)
    y_pred_rounded = np.round(y_pred, decimals=decimals)

    # Compute accuracy as proportion of exact matches
    matches = y_true_rounded == y_pred_rounded
    accuracy = np.mean(matches)

    return accuracy

# Dictionary to map metric names to their corresponding functions
metrics_map = {
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_error": mean_squared_error,
    "weighted_accuracy": weighted_accuracy,
    "accuracy": accuracy_score,
    "rounded_accuracy": rounded_accuracy
}



def compare_predictions(
    true_values_path, predictions_path, error_threshold, metric, target_col, id_col='id', is_lower=True
):
    try:
        y_true = pd.read_csv(true_values_path)
    except FileNotFoundError:
        print(f"Error: The file {true_values_path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {true_values_path} is empty or not well formatted.")
        sys.exit(1)

    try:
        y_pred = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"Error: The file {predictions_path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {predictions_path} is empty or not well formatted.")
        sys.exit(1)

    # Verify that both files contain the required 'id_col' column
    if id_col not in y_true.columns or id_col not in y_pred.columns:
        print(f"Error: Missing the '{id_col}' column.")
        sys.exit(1)

    # Merge datasets to ensure they are comparable
    try:
        merged_data = pd.merge(y_true, y_pred, on=id_col, suffixes=("_true", "_pred"))
    except KeyError:
        print(f"Error: {id_col} mismatch between the true values and predictions.")
        sys.exit(1)
    # Check if the target column exists and is numeric
    if f"{target_col}_pred" not in merged_data.columns:
        print(f"Error: The target column '{target_col}' does not exist in your file.")
        sys.exit(1)

    # Ensure the target column contains numeric data for both true and predicted values
    if not pd.api.types.is_numeric_dtype(
        merged_data[f"{target_col}_true"]
    ) or not pd.api.types.is_numeric_dtype(merged_data[f"{target_col}_pred"]):
        print(
            f"Error: The target column '{target_col}' must contain numeric data in both true values and predictions."
        )
        sys.exit(1)

    try:
        score = metrics_map[metric](
            merged_data[f"{target_col}_true"], merged_data[f"{target_col}_pred"]
        )
    except KeyError:
        print(f"Error: The metric '{metric}' is not supported.")
        sys.exit(1)

    # Check if the score exceeds the threshold
    if is_lower:
        if score > error_threshold:
            print(f"Error: {metric} score: {score} exceeds threshold {error_threshold}.")
            sys.exit(1)
        else:
            print(f"Success: {metric} score: {score} is within the acceptable threshold.")
    else:
        if score < error_threshold:
            print(f"Error: {metric} score: {score} bellow threshold {error_threshold}.")
            sys.exit(1)
        else:
            print(f"Success: {metric} score: {score} is within the acceptable threshold.")

if __name__ == "__main__":
    true_values_path = sys.argv[1]
    predictions_path = sys.argv[2]
    error_threshold = float(sys.argv[3])
    metric = sys.argv[4]
    target_col = sys.argv[5]
    
    # Default the id_col to 'id' if not provided
    id_col = sys.argv[6] if len(sys.argv) > 6 else 'id'
    
    is_lower_str = sys.argv[7] if len(sys.argv) > 7 else 'true'
    is_lower = is_lower_str.lower() == 'true'

    compare_predictions(
        true_values_path, predictions_path, error_threshold, metric, target_col, id_col, is_lower
    )