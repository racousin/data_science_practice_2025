#!/bin/bash

# Usage: ./exercise1.sh <username> <current_utc_time> <aws-access-key-id> <aws-secret-access-key> <aws-region>

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <username> <current_utc_time> <aws-access-key-id> <aws-secret-access-key> <aws-region>"
  exit 1
fi

USERNAME=$1
CURRENT_UTC_TIME=$2
AWS_ACCESS_KEY_ID=$3
AWS_SECRET_ACCESS_KEY=$4
AWS_DEFAULT_REGION=$5
MODULE_NUMBER="8"
TARGET_PATH="module${MODULE_NUMBER}/test_target.csv"
PREDICTIONS_PATH="${USERNAME}/module${MODULE_NUMBER}/submission.csv"
RESULTS_PATH="test_target.csv"
RESULTS_DIR="./results"  # Directory to store results
RESULT_FILE="${RESULTS_DIR}/module${MODULE_NUMBER}_exercise1.json"  # File to store this exercise's results
IS_LOWER=false

mkdir -p $RESULTS_DIR  # Ensure results directory exists

# Setup a Python virtual environment
# python3 -m venv venv
source venv/bin/activate

# Download test_target.csv from S3 using a provided script
python tests/utils/download_from_s3.py $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION $TARGET_PATH $RESULTS_PATH

# Run comparison using a provided Python script
# Using rounded_accuracy metric with 2 decimal precision
set +e
ACCURACY_THRESHOLD=0.70
METRIC="rounded_accuracy"
TARGET_COL="solution"
ID_COL="id"
COMPARE_OUTPUT=$(python tests/utils/compare_predictions.py $RESULTS_PATH $PREDICTIONS_PATH $ACCURACY_THRESHOLD $METRIC $TARGET_COL $ID_COL $IS_LOWER 2>&1)
COMPARE_EXIT_CODE=$?
SCORE=$(echo "$COMPARE_OUTPUT" | sed -n 's/.*score: \([0-9.]*\).*/\1/p')
set -e

# Deactivate the virtual environment
deactivate

# Prepare the output in JSON format based on the test results
if [ "$COMPARE_EXIT_CODE" -eq 0 ]; then
    jq -n \
        --arg is_passed "true" \
        --arg score "$SCORE" \
        --arg logs "$COMPARE_OUTPUT" \
        --arg time "$CURRENT_UTC_TIME" \
        '{is_passed_test: ($is_passed == "true"), score: $score, logs: $logs, updated_time_utc: $time}' > $RESULT_FILE
else
    # Extract only the score if present, else default to "0"
    SCORE=$(echo "$COMPARE_OUTPUT" | sed -n 's/.*score: \([0-9.]*\).*/\1/p')
    if [ -z "$SCORE" ]; then
        SCORE="0"
    fi
    jq -n \
        --arg is_passed "false" \
        --arg score "$SCORE" \
        --arg logs "$COMPARE_OUTPUT" \
        --arg time "$CURRENT_UTC_TIME" \
        '{is_passed_test: ($is_passed == "true"), score: $score, logs: $logs, updated_time_utc: $time}' > $RESULT_FILE
fi

# Clean up the virtual environment directory
# rm -rf venv
