#!/bin/bash

# Usage: ./exercise1.sh <username> <aws-access-key-id> <aws-secret-access-key> <aws-region>

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <username> <current_utc_time> <aws-access-key-id> <aws-secret-access-key> <aws-region>"
  exit 1
fi

USERNAME=$1
CURRENT_UTC_TIME=$2
AWS_ACCESS_KEY_ID=$3
AWS_SECRET_ACCESS_KEY=$4
AWS_DEFAULT_REGION=$5
MODULE_NUMBER="7"
TARGET_PATH="module${MODULE_NUMBER}/y_test_target.pkl"
PREDICTIONS_PATH="${USERNAME}/module${MODULE_NUMBER}/predictions.csv"
RESULTS_PATH="y_test_target.pkl"
RESULTS_DIR="./results"  # Directory to store results
RESULT_FILE="${RESULTS_DIR}/module${MODULE_NUMBER}_exercise1.json"  # File to store this exercise's results

mkdir -p $RESULTS_DIR  # Ensure results directory exists

# Setup a Python virtual environment
# python3 -m venv venv
source venv/bin/activate

# Download y_test_target.pkl from S3 using a provided script
python tests/utils/download_from_s3.py $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION $TARGET_PATH $RESULTS_PATH

# Run mAP computation using the compute_map script
set +e
MAP_THRESHOLD=0.70
COMPARE_OUTPUT=$(python tests/utils/compute_map.py $RESULTS_PATH $PREDICTIONS_PATH $MAP_THRESHOLD 2>&1)
COMPARE_EXIT_CODE=$?
# Extract score using sed (compatible with macOS)
SCORE=$(echo "$COMPARE_OUTPUT" | sed -n 's/.*mAP50: \([0-9.]*\).*/\1/p' | head -1)
set -e

# Deactivate the virtual environment
deactivate

# Escape newlines and quotes in logs for JSON
ESCAPED_LOGS=$(echo "$COMPARE_OUTPUT" | python -c "import sys, json; print(json.dumps(sys.stdin.read())[1:-1])")

# Prepare the output in JSON format based on the test results
if [ "$COMPARE_EXIT_CODE" -eq 0 ]; then
    echo "{\"is_passed_test\": true, \"score\": \"$SCORE\", \"logs\": \"${ESCAPED_LOGS}\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
else
    # Extract only the score if present, else default to "0"
    if [ -z "$SCORE" ]; then
        SCORE="0"
    fi
    echo "{\"is_passed_test\": false, \"score\": \"$SCORE\", \"logs\": \"${ESCAPED_LOGS}\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
fi

# Clean up the virtual environment directory
# rm -rf venv
