#!/bin/bash
# Computes the progress and error percentages based on passed exercises and updates the central configuration

USER=$1
GITHUB_REPOSITORY_NAME=$2
AWS_ACCESS_KEY_ID=$3
AWS_SECRET_ACCESS_KEY=$4
AWS_DEFAULT_REGION=$5

# Configure AWS CLI
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

# Download the individual student's JSON file, create from template if it doesn't exist
if ! aws s3 cp s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/"$USER".json "$USER".json 2>/dev/null; then
  echo "Student JSON file doesn't exist, creating from template..."
  cp ./scripts/student.json "$USER".json
  aws s3 cp "$USER".json s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/"$USER".json
fi

# Calculate the progress percentage
TOTAL_EXERCISES=$(jq '.[] | .[] | .is_passed_test' "$USER".json | wc -l)
PASSED_EXERCISES=$(jq '.[] | .[] | select(.is_passed_test == true) | .is_passed_test' "$USER".json | wc -l)

if [ "$TOTAL_EXERCISES" -eq 0 ]; then
  PROGRESS=0
else
  PROGRESS=$(echo "scale=2; $PASSED_EXERCISES / $TOTAL_EXERCISES" | bc)
fi

# Calculate the error percentage as ($UPDATED_EXERCISES - $PASSED_EXERCISES) / $TOTAL_EXERCISES
UPDATED_EXERCISES=$(jq '.[] | .[] | select(.updated_time_utc != "") | .updated_time_utc' "$USER".json | wc -l)

if [ "$TOTAL_EXERCISES" -eq 0 ]; then
  ERROR_PERCENTAGE=0
else
  ERROR_PERCENTAGE=$(echo "scale=2; ($UPDATED_EXERCISES - $PASSED_EXERCISES) / $TOTAL_EXERCISES" | bc)
fi

# Acquire lock for the central configuration file
LOCK_FILE=s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/lock.txt
while aws s3 ls "$LOCK_FILE" > /dev/null 2>&1; do
  echo "Waiting for lock to release..."
  sleep 1
done
echo "Acquiring lock..."
echo "Lock" | aws s3 cp - "$LOCK_FILE"

# Update the central configuration
aws s3 cp s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/students.json students.json
jq --arg user "$USER" --argjson progress "$PROGRESS" --argjson error_percentage "$ERROR_PERCENTAGE" '(.[$user].progress_percentage) = $progress | (.[$user].error_percentage) = $error_percentage' students.json > updated_students.json
mv updated_students.json students.json
aws s3 cp students.json s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/students.json

# Release the lock
aws s3 rm "$LOCK_FILE"

# Clean up local files
# rm "$USER".json students.json
