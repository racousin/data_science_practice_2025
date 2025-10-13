"""
Compute mAP (mean Average Precision) for object detection
Supports YOLO format predictions
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes
    Boxes in format: [x_center, y_center, width, height] (normalized 0-1)
    """
    # Convert from center format to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    # Compute union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def compute_ap(recalls, precisions):
    """
    Compute Average Precision (AP) using the 11-point interpolation
    """
    # Add sentinel values at the beginning and end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Compute AP by integrating the precision-recall curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def compute_map_for_class(ground_truths, predictions, iou_threshold=0.5):
    """
    Compute mAP for a single class

    Args:
        ground_truths: List of ground truth boxes for each image [[boxes], [boxes], ...]
                      Each box: [class_id, x_center, y_center, width, height]
        predictions: List of predicted boxes for each image [[boxes], [boxes], ...]
                    Each box: [class_id, confidence, x_center, y_center, width, height]
        iou_threshold: IoU threshold for matching predictions to ground truth

    Returns:
        mAP score (0-1)
    """
    # Flatten all ground truths and predictions with image indices
    all_gt = []
    all_pred = []

    for img_idx, (gt_boxes, pred_boxes) in enumerate(zip(ground_truths, predictions)):
        # Process ground truths
        for box in gt_boxes:
            if len(box) >= 5:  # class_id, x, y, w, h
                all_gt.append({
                    'image_id': img_idx,
                    'box': box[1:5]  # x, y, w, h
                })

        # Process predictions
        for box in pred_boxes:
            if len(box) >= 6:  # class_id, confidence, x, y, w, h
                all_pred.append({
                    'image_id': img_idx,
                    'confidence': box[1],
                    'box': box[2:6]  # x, y, w, h
                })

    if len(all_gt) == 0:
        print("Warning: No ground truth boxes found")
        return 0.0

    if len(all_pred) == 0:
        print("Warning: No predicted boxes found")
        return 0.0

    # Sort predictions by confidence (descending)
    all_pred = sorted(all_pred, key=lambda x: x['confidence'], reverse=True)

    # Track which ground truths have been matched
    gt_matched = [False] * len(all_gt)

    true_positives = []
    false_positives = []

    # For each prediction
    for pred in all_pred:
        # Find ground truths from the same image
        max_iou = 0
        max_gt_idx = -1

        for gt_idx, gt in enumerate(all_gt):
            if gt['image_id'] == pred['image_id'] and not gt_matched[gt_idx]:
                iou = compute_iou(pred['box'], gt['box'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

        # Check if this is a true positive or false positive
        if max_iou >= iou_threshold and max_gt_idx >= 0:
            gt_matched[max_gt_idx] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    # Compute cumulative sums
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)

    # Compute precision and recall
    recalls = tp_cumsum / len(all_gt)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP
    ap = compute_ap(recalls, precisions)

    return ap


def load_predictions_from_csv(csv_path, num_images):
    """
    Load predictions from CSV file
    CSV format: image_id, box_idx, class_id, confidence, x_center, y_center, width, height
    """
    predictions = [[] for _ in range(num_images)]

    try:
        df = pd.read_csv(csv_path)

        if 'image_id' not in df.columns:
            print("Error: CSV must contain 'image_id' column")
            return None

        required_cols = ['image_id', 'class_id', 'confidence', 'x_center', 'y_center', 'width', 'height']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: CSV must contain '{col}' column")
                return None

        # Group by image
        for img_id in range(num_images):
            img_preds = df[df['image_id'] == img_id]
            boxes = []
            for _, row in img_preds.iterrows():
                boxes.append([
                    row['class_id'],
                    row['confidence'],
                    row['x_center'],
                    row['y_center'],
                    row['width'],
                    row['height']
                ])
            predictions[img_id] = np.array(boxes) if boxes else np.array([])

    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None

    return predictions


def main(ground_truth_path, predictions_path, threshold):
    """
    Main function to compute mAP

    Args:
        ground_truth_path: Path to ground truth pickle file (y_test_target.pkl)
        predictions_path: Path to predictions CSV file
        threshold: Minimum mAP50 score required to pass
    """
    # Load ground truth
    try:
        with open(ground_truth_path, 'rb') as f:
            ground_truths = pickle.load(f)
        print(f"Loaded {len(ground_truths)} ground truth images")
    except Exception as e:
        print(f"Error: Failed to load ground truth: {e}")
        sys.exit(1)

    # Load predictions
    predictions = load_predictions_from_csv(predictions_path, len(ground_truths))
    if predictions is None:
        sys.exit(1)

    print(f"Loaded predictions for {len(predictions)} images")

    # Compute mAP50 (IoU threshold = 0.5)
    map50 = compute_map_for_class(ground_truths, predictions, iou_threshold=0.5)

    print(f"\n=== Results ===")
    print(f"mAP50: {map50:.4f}")
    print(f"Threshold: {threshold:.4f}")

    if map50 >= threshold:
        print(f"Success: mAP50 score: {map50:.4f} meets threshold {threshold:.4f}")
        sys.exit(0)
    else:
        print(f"Error: mAP50 score: {map50:.4f} below threshold {threshold:.4f}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compute_map.py <ground_truth_pkl> <predictions_csv> <threshold>")
        sys.exit(1)

    ground_truth_path = sys.argv[1]
    predictions_path = sys.argv[2]
    threshold = float(sys.argv[3])

    main(ground_truth_path, predictions_path, threshold)
