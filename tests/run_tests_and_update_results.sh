#!/bin/bash
# This script is simplified to handle running all test scripts and using Python to aggregate and upload results.

USER=$1
AWS_ACCESS_KEY_ID=$2
AWS_SECRET_ACCESS_KEY=$3
AWS_DEFAULT_REGION=$4
CHANGED_MODULES=$5
GITHUB_REPOSITORY_NAME=$6


# Directory where test scripts are located
TEST_DIR="./tests"
RESULTS_DIR="./results"
FINAL_JSON="$RESULTS_DIR/${USER}_final_results.json"
S3_BUCKET="www.raphaelcousin.com"
S3_KEY="repositories/$GITHUB_REPOSITORY_NAME/students/$USER.json"

# Get current UTC time in ISO 8601 format
CURRENT_UTC_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")

# Load AWS credentials
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

mkdir -p $RESULTS_DIR  # Ensure results directory exists

# Setup Python environment before running tests
python3 -m venv venv || { echo "ERROR: Failed to create virtual environment"; exit 1; }
source venv/bin/activate || { echo "ERROR: Failed to activate virtual environment"; exit 1; }
pip install -r tests/requirements.txt || { echo "ERROR: Failed to install Python dependencies"; exit 1; }

echo $CHANGED_MODULES
IFS=' ' read -r -a modules <<< "$CHANGED_MODULES"  # Split CHANGED_MODULES into an array
for module in "${modules[@]}"; do
    MODULE_DIR="${TEST_DIR}/data-science-practice/module${module}"
    if [ -d "$MODULE_DIR" ]; then
        echo "Processing module $module..."
        EXERCISE_SCRIPTS=$(find "$MODULE_DIR" -name '*.sh')

        for script in $EXERCISE_SCRIPTS; do
            exercise=$(basename "${script%.*}")  # Remove file extension

            echo "Running tests for module $module, exercise $exercise..."
            echo $USER $CURRENT_UTC_TIME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION
            "$script" $USER $CURRENT_UTC_TIME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION || {
                echo "ERROR: Test script failed: $script";
                deactivate || true;
                rm -rf venv || true;
                rm -rf $RESULTS_DIR || true;
                exit 1;
            }
        done
    else
        echo "No tests found for module $module."
    fi
done

python tests/aggregate_results.py $RESULTS_DIR $FINAL_JSON $S3_BUCKET $S3_KEY $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION || {
    echo "ERROR: Failed to aggregate results and upload to S3";
    deactivate || true;
    rm -rf venv || true;
    rm -rf $RESULTS_DIR || true;
    exit 1;
}

echo "SUCCESS: All tests completed and results uploaded to S3"
deactivate || true
rm -rf venv || true
rm -rf $RESULTS_DIR || true
