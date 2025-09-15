#!/bin/bash

# This script runs tests for module1 and expects a username as a parameter
# Usage: ./exercise1.sh <username>

USERNAME=$1
CURRENT_UTC_TIME=$2
MODULE_NUMBER="1"  # Since this script is specifically for module1, we can hardcode the module number.
RESULTS_DIR="./results"  # Directory to store results
RESULT_FILE="${RESULTS_DIR}/module${MODULE_NUMBER}_exercise1.json"  # File to store this exercise's results

mkdir -p $RESULTS_DIR  # Ensure results directory exists

echo "Starting tests for module1 for user $USERNAME..."

FILE_PATH="${USERNAME}/module${MODULE_NUMBER}/user"

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
  # Check if student created file with .txt extension (common mistake)
  if [ -f "${FILE_PATH}.txt" ]; then
    echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"Error: Found ${FILE_PATH}.txt but expected ${FILE_PATH} (without .txt extension). Please rename your file to remove the .txt extension.\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
    exit 1
  fi
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"Error: File $FILE_PATH does not exist.\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# Read the file content and remove non-printable characters including BOM
FILE_CONTENT=$(cat "$FILE_PATH" | tr -cd '[:print:]' | xargs)
IFS=',' read -r -a content_array <<< "$FILE_CONTENT"

# Debugging: Print the extracted content and the username
echo "DEBUG: Username expected: '$USERNAME'"
echo "DEBUG: Username in file: '${content_array[0]}'"
echo "DEBUG: Full file content: '$FILE_CONTENT'"

# Check if the first element in the file (the username) matches the expected username
if [[ "${content_array[0]}" != "$USERNAME" ]]; then
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"Error: Username mismatch in $FILE_PATH. Received: '${content_array[0]}', Expected: '$USERNAME'\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# Check if the file contains exactly two commas
comma_count=$(grep -o ',' <<< "$FILE_CONTENT" | wc -l)
if [[ "$comma_count" -ne 2 ]]; then
  echo "{\"is_passed_test\": false, \"score\": \"0\", \"logs\": \"Error: Incorrect number of commas in $FILE_PATH. Received: '$comma_count' commas, Expected: 2 commas. File content: '$FILE_CONTENT'\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
  exit 1
fi

# If the file exists and content is correct, output success message in JSON format
echo "{\"is_passed_test\": true, \"score\": \"100\", \"logs\": \"module${MODULE_NUMBER} tests passed successfully for ${USERNAME}.\", \"updated_time_utc\": \"$CURRENT_UTC_TIME\"}" > $RESULT_FILE
