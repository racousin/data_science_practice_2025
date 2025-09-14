#!/bin/bash

# Script to detect the actual author of a PR/merge commit
# Handles multiple edge cases including:
# - PR merged by someone else
# - Merge commits created by GitHub
# - Branch not created by the final author
# - Direct pushes to main

set -e

# Get commit hash (default to HEAD if not provided)
COMMIT_HASH=${1:-$(git rev-parse HEAD)}

# Get repository info
REPO=$(basename `git rev-parse --show-toplevel`)
OWNER=$(git remote get-url origin | sed -n 's#.*github.com[:/]\(.*\)/.*#\1#p')

echo "========================================="
echo "Analyzing commit: $COMMIT_HASH"
echo "Repository: $OWNER/$REPO"
echo "========================================="

# Check if we have a GitHub token
if [ -n "$TOKEN_USER_READ" ]; then
    AUTH_HEADER="Authorization: token $TOKEN_USER_READ"
elif [ -n "$GITHUB_TOKEN" ]; then
    AUTH_HEADER="Authorization: token $GITHUB_TOKEN"
else
    echo "Warning: No GitHub token found. API calls may be rate-limited."
    AUTH_HEADER=""
fi

# Function to call GitHub API
github_api() {
    local url=$1
    if [ -n "$AUTH_HEADER" ]; then
        curl -s -H "$AUTH_HEADER" "$url"
    else
        curl -s "$url"
    fi
}

# Check if this is a merge commit
PARENT_COUNT=$(git show --format=%P -s $COMMIT_HASH | wc -w)

if [ "$PARENT_COUNT" -eq 2 ]; then
    echo "Type: Merge commit"
    echo ""

    # Strategy 1: Extract PR number from commit message
    PR_NUMBER=$(git log -1 --format=%B $COMMIT_HASH | grep -oE '#[0-9]+' | head -1 | tr -d '#')

    if [ -n "$PR_NUMBER" ]; then
        echo "Found PR reference: #$PR_NUMBER"
        PR_API_URL="https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER"
        PR_DETAILS=$(github_api "$PR_API_URL")

        # Get PR author
        PR_AUTHOR=$(echo "$PR_DETAILS" | jq -r '.user.login // empty')

        # Get PR merger (who clicked the merge button)
        PR_MERGER=$(echo "$PR_DETAILS" | jq -r '.merged_by.login // empty')

        # Get branch name
        PR_BRANCH=$(echo "$PR_DETAILS" | jq -r '.head.ref // empty')

        echo "PR Author (opened PR): $PR_AUTHOR"
        echo "PR Merger (clicked merge): $PR_MERGER"
        echo "Branch name: $PR_BRANCH"
        echo ""

        if [ -n "$PR_AUTHOR" ] && [ "$PR_AUTHOR" != "null" ]; then
            FINAL_AUTHOR="$PR_AUTHOR"
            echo "✓ Using PR author as the final author: $FINAL_AUTHOR"
        fi
    fi

    # Strategy 2: Analyze the merged branch
    if [ -z "$FINAL_AUTHOR" ] || [ "$FINAL_AUTHOR" = "null" ]; then
        echo "Analyzing merged branch commits..."

        # Get both parents
        PARENT1=$(git show --format=%P -s $COMMIT_HASH | awk '{print $1}')
        PARENT2=$(git show --format=%P -s $COMMIT_HASH | awk '{print $2}')

        echo "Main branch parent: $PARENT1"
        echo "Merged branch HEAD: $PARENT2"

        # Get merge base
        MERGE_BASE=$(git merge-base $PARENT1 $PARENT2)
        echo "Merge base: $MERGE_BASE"
        echo ""

        # List all commits in the merged branch
        echo "Commits in merged branch:"
        git log $MERGE_BASE..$PARENT2 --format='  %h - %an <%ae> - %s'
        echo ""

        # Find the most frequent author
        BRANCH_AUTHORS=$(git log $MERGE_BASE..$PARENT2 --format='%an' | sort | uniq -c | sort -rn)
        echo "Author frequency in branch:"
        echo "$BRANCH_AUTHORS" | sed 's/^/  /'
        echo ""

        MOST_FREQUENT_AUTHOR=$(echo "$BRANCH_AUTHORS" | head -1 | awk '{$1=""; print $0}' | xargs)
        echo "Most frequent author: $MOST_FREQUENT_AUTHOR"

        # Try to get GitHub username for the branch HEAD commit
        COMMIT_API_URL="https://api.github.com/repos/$OWNER/$REPO/commits/$PARENT2"
        COMMIT_DETAILS=$(github_api "$COMMIT_API_URL")
        GITHUB_USERNAME=$(echo "$COMMIT_DETAILS" | jq -r '.author.login // empty')

        if [ -n "$GITHUB_USERNAME" ] && [ "$GITHUB_USERNAME" != "null" ]; then
            FINAL_AUTHOR="$GITHUB_USERNAME"
            echo "✓ Found GitHub username from branch HEAD: $FINAL_AUTHOR"
        else
            FINAL_AUTHOR="$MOST_FREQUENT_AUTHOR"
            echo "⚠ Using git author name (no GitHub username found): $FINAL_AUTHOR"
        fi
    fi

else
    echo "Type: Regular commit (not a merge)"
    echo ""

    # Get commit details
    COMMIT_AUTHOR=$(git show -s --format='%an' $COMMIT_HASH)
    COMMIT_EMAIL=$(git show -s --format='%ae' $COMMIT_HASH)

    echo "Git author: $COMMIT_AUTHOR <$COMMIT_EMAIL>"

    # Try to get GitHub username
    COMMIT_API_URL="https://api.github.com/repos/$OWNER/$REPO/commits/$COMMIT_HASH"
    COMMIT_DETAILS=$(github_api "$COMMIT_API_URL")
    GITHUB_USERNAME=$(echo "$COMMIT_DETAILS" | jq -r '.author.login // empty')

    if [ -n "$GITHUB_USERNAME" ] && [ "$GITHUB_USERNAME" != "null" ]; then
        FINAL_AUTHOR="$GITHUB_USERNAME"
        echo "✓ Found GitHub username: $FINAL_AUTHOR"
    else
        FINAL_AUTHOR="$COMMIT_AUTHOR"
        echo "⚠ Using git author name (no GitHub username found): $FINAL_AUTHOR"
    fi
fi

echo ""
echo "========================================="
echo "FINAL DETECTED AUTHOR: $FINAL_AUTHOR"
echo "========================================="

# Export for use in scripts
export DETECTED_AUTHOR="$FINAL_AUTHOR"