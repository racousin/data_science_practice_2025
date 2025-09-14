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

sudo apt-get install -y xmlstarlet

# Setup a Python virtual environment
# python3 -m venv venv
source venv/bin/activate

PACKAGE_DIR="$(pwd)/${USERNAME}/module2/mysupertools"
if [ ! -d "$PACKAGE_DIR" ]; then
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"Package directory not found at $PACKAGE_DIR.\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# Attempt to install the package using pip
if ! pip install $PACKAGE_DIR; then
  LOGS="Failed to install package from $PACKAGE_DIR."
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"$LOGS\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

export PYTHONPATH=$PACKAGE_DIR:$PYTHONPATH

if ! python -m pytest $TESTS_DIR/test_exercise1.py --junitxml=results.xml; then
  # Check if results.xml exists and has content
  if [ -f results.xml ] && [ -s results.xml ]; then
    # Extract error messages from the XML file using xmlstarlet
    ERROR_DETAILS=$(xmlstarlet sel -t -m "//error | //failure" -v . -n results.xml | \
            sed ':a;N;$!ba;s/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/&gt;/>/g; s/&lt;/</g; s/&amp;/&/g' | \
            tr -d '\r' | \
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
