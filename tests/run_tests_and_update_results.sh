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

# Aggregate results and upload to S3
python3 -m venv venv
source venv/bin/activate
pip install -r tests/requirements.txt

echo $CHANGED_MODULES
IFS=' ' read -r -a modules <<< "$CHANGED_MODULES"  # Split CHANGED_MODULES into an array
for module in "${modules[@]}"; do
    MODULE_DIR="${TEST_DIR}/module${module}"
    if [ -d "$MODULE_DIR" ]; then
        echo "Processing module $module..."
        EXERCISE_SCRIPTS=$(find "$MODULE_DIR" -name '*.sh')

        for script in $EXERCISE_SCRIPTS; do
            exercise=$(basename "${script%.*}")  # Remove file extension

            echo "Running tests for module $module, exercise $exercise..."
            echo $USER $CURRENT_UTC_TIME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION
            "$script" $USER $CURRENT_UTC_TIME $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION
        done
    else
        echo "No tests found for module $module."
    fi
done

python tests/aggregate_results.py $RESULTS_DIR $FINAL_JSON $S3_BUCKET $S3_KEY $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY $AWS_DEFAULT_REGION
deactivate
# rm -rf venv
# Cleanup local results
rm -rf $RESULTS_DIR
