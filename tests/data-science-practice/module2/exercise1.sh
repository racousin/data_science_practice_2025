#!/bin/bash

# This script runs tests for module2 and expects a username and AWS credentials
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
MODULE_NUMBER="2"
RESULTS_DIR="./results"  # Directory to store results
RESULT_FILE="${RESULTS_DIR}/module${MODULE_NUMBER}_exercise1.json"  # File to store this exercise's results
TESTS_DIR="./tests/data-science-practice/module2"

mkdir -p $RESULTS_DIR  # Ensure results directory exists

# Install xmlstarlet if not available (macOS users should use: brew install xmlstarlet)
if ! command -v xmlstarlet &> /dev/null; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y xmlstarlet
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Please install xmlstarlet using: brew install xmlstarlet"
        exit 1
    fi
fi

# Setup a Python virtual environment
python3 -m venv venv
source venv/bin/activate

PACKAGE_DIR="$(pwd)/${USERNAME}/module2/mysupertools"
if [ ! -d "$PACKAGE_DIR" ]; then
  LOGS="No folder module2/mysupertools/"
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"$LOGS\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# Check for build artifacts that should not be committed
FOUND_ARTIFACTS=""

# Check for build/dist directories
if [ -d "$PACKAGE_DIR/build" ]; then
  FOUND_ARTIFACTS="${FOUND_ARTIFACTS}build/ "
fi
if [ -d "$PACKAGE_DIR/dist" ]; then
  FOUND_ARTIFACTS="${FOUND_ARTIFACTS}dist/ "
fi

# Check for .egg-info directories
EGG_INFO_DIRS=$(find "$PACKAGE_DIR" -name "*.egg-info" -type d 2>/dev/null)
if [ -n "$EGG_INFO_DIRS" ]; then
  FOUND_ARTIFACTS="${FOUND_ARTIFACTS}$(basename $EGG_INFO_DIRS) "
fi

# Check for __pycache__ directories
PYCACHE_DIRS=$(find "$PACKAGE_DIR" -name "__pycache__" -type d 2>/dev/null)
if [ -n "$PYCACHE_DIRS" ]; then
  FOUND_ARTIFACTS="${FOUND_ARTIFACTS}__pycache__/ "
fi

# Check for .pyc files
PYC_FILES=$(find "$PACKAGE_DIR" -name "*.pyc" -type f 2>/dev/null)
if [ -n "$PYC_FILES" ]; then
  FOUND_ARTIFACTS="${FOUND_ARTIFACTS}*.pyc "
fi

if [ -n "$FOUND_ARTIFACTS" ]; then
  LOGS="Build artifacts found that should not be committed: ${FOUND_ARTIFACTS}. Remove them from the repository in a new branch and try again."
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"$LOGS\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# Attempt to install the package using pip
if ! pip install $PACKAGE_DIR; then
  LOGS="Failed to install package from ${USERNAME}/module2/mysupertools. Expected structure:\\n${USERNAME}/module2/mysupertools/\\n    ├── pyproject.toml\\n    └── mysupertools/\\n        ├── __init__.py\\n        └── tool/\\n            ├── __init__.py\\n            └── operation_a_b.py\\n\\nNote: pyproject.toml should be in module2/mysupertools/"
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"$LOGS\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

export PYTHONPATH=$PACKAGE_DIR:$PYTHONPATH

if ! python -m pytest $TESTS_DIR/test_exercise1.py --junitxml=results.xml; then
  # Check if results.xml exists and has content
  if [ -f results.xml ] && [ -s results.xml ]; then
    # Extract error messages from the XML file using xmlstarlet
    ERROR_DETAILS=$(xmlstarlet sel -t -m "//error | //failure" -v . -n results.xml | \
            sed 's/\\/\\\\/g; s/"/\\"/g; s/&gt;/>/g; s/&lt;/</g; s/&amp;/&/g' | \
            tr -d '\r\n' | \
            awk '{gsub(/[[:cntrl:]]/, ""); print}')

  else
    ERROR_DETAILS="No detailed error information was found, or the results file was not created."
  fi
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"$ERROR_DETAILS\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
else
  echo "{\"is_passed_test\": true, \"score\": \"100\", \"logs\": \"All tests passed successfully for module2.\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
fi



# TODO remove the pkg installed and be sure we don't loose the current pkg
# pip uninstall $PACKAGE_DIR
pip uninstall -y mysupertools
pip install -r tests/data-science-practice/requirements.txt

# Deactivate the virtual environment
deactivate

# Clean up the virtual environment directory
# rm -rf venv

# Exit with the result code from pytest to indicate success or failure in an automated system
exit $RESULT_CODE
