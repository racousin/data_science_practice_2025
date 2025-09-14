#!/bin/bash
# Script to check and add a user to the student list

USER=$1
GITHUB_REPOSITORY_NAME=$2
AWS_ACCESS_KEY_ID=$3
AWS_SECRET_ACCESS_KEY=$4
AWS_DEFAULT_REGION=$5

aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

# Check if students.json exists on S3, create empty dict if not
if ! aws s3 ls s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/students.json > /dev/null 2>&1; then
  echo "students.json does not exist. Creating empty dict..."
  echo "{}" > students.json
  aws s3 cp students.json s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/students.json
  echo "Created empty students.json on S3."
else
  aws s3 cp s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/students.json students.json
fi

if jq -e 'has("'"$USER"'")' students.json; then
  echo "$USER is already in the list."
else
  echo "$USER needs to be added to the list. Acquiring lock..."
  LOCK_EXISTS=$(aws s3 ls s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/lock.txt && echo "exists" || echo "not exists")
  while [ "$LOCK_EXISTS" = "exists" ]; do
    echo "Waiting for lock to release..."
    sleep 1
    LOCK_EXISTS=$(aws s3 ls s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/lock.txt && echo "exists" || echo "not exists")
  done
  echo "Creating lock..."
  echo "Lock" | aws s3 cp - s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/lock.txt

  jq '."'"$USER"'" = {"progress_percentage": 0, "error_percentage": 0}' students.json > updated_students.json
  mv updated_students.json students.json
  aws s3 cp students.json s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/students.json

  echo "Releasing lock..."
  aws s3 rm s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/config/lock.txt
fi

# Always ensure the individual student JSON file exists
if ! aws s3 ls s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/"$USER".json > /dev/null 2>&1; then
  echo "Creating initial student JSON for $USER."
  if [ -f "./scripts/student.json" ]; then
    aws s3 cp ./scripts/student.json s3://www.raphaelcousin.com/repositories/$GITHUB_REPOSITORY_NAME/students/"$USER".json
  else
    echo "ERROR: ./scripts/student.json not found. Current directory: $(pwd)"
    echo "Files in scripts/:"
    ls -la scripts/ || echo "scripts/ directory not found"
    exit 1
  fi
else
  echo "Student JSON file for $USER already exists."
fi
