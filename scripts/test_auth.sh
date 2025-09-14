#!/bin/bash

# Check if a commit hash is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <commit_hash>"
  exit 1
fi

# Set the commit hash to test
COMMIT_HASH=$1

# GitHub Token for user read should be set as an environment variable
GITHUB_TOKEN_USER_READ=${GITHUB_TOKEN_USER_READ:-""}

# Check if GITHUB_TOKEN_USER_READ is set
if [ -z "$GITHUB_TOKEN_USER_READ" ]; then
  echo "Error: GITHUB_TOKEN_USER_READ environment variable is not set."
  exit 1
fi

# Get repository info from the current directory
REPO=$(basename `git rev-parse --show-toplevel`)
OWNER=$(git remote get-url origin | sed -n 's#.*github.com[:/]\(.*\)/.*#\1#p')

# Use the GitHub Commit API to get commit details
echo "Fetching GitHub username for commit: $COMMIT_HASH"
COMMIT_API_URL="https://api.github.com/repos/$OWNER/$REPO/commits/$COMMIT_HASH"
COMMIT_DETAILS=$(curl -s -H "Authorization: token $GITHUB_TOKEN_USER_READ" $COMMIT_API_URL)

# Extract the commit author's GitHub username from the API response
USER_LOGIN=$(echo "$COMMIT_DETAILS" | jq -r '.author.login')

# Output the detected GitHub username
echo "Detected GitHub Username: $USER_LOGIN"

# Check if the username was successfully determined
if [ -n "$USER_LOGIN" ] && [ "$USER_LOGIN" != "null" ]; then
  echo "GitHub Username successfully determined: $USER_LOGIN"
else
  echo "Failed to determine GitHub username for commit $COMMIT_HASH"
  exit 1
fi
