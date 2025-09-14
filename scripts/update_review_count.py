#!/usr/bin/env python3
"""
Update review count for a student in the students.json file stored in S3.
This script is used by GitHub Actions to track PR reviews.
"""

import json
import sys
import argparse
import boto3
from datetime import datetime
from typing import Dict, Any, Optional


def download_students_json(s3_client, bucket: str, key: str) -> Dict[str, Any]:
    """Download students.json from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"Error downloading students.json: {e}")
        sys.exit(1)


def upload_students_json(s3_client, bucket: str, key: str, data: Dict[str, Any]) -> None:
    """Upload updated students.json to S3."""
    try:
        json_content = json.dumps(data, indent=2)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json_content,
            ContentType='application/json'
        )
        print(f"Successfully uploaded updated students.json to s3://{bucket}/{key}")
    except Exception as e:
        print(f"Error uploading students.json: {e}")
        sys.exit(1)


def find_student_by_github_username(students: Dict[str, Any], github_username: str) -> Optional[tuple[str, Dict[str, Any]]]:
    """Find a student record by GitHub username."""
    github_username_lower = github_username.lower()
    for student_id, data in students.items():
        if data.get('github_username', '').lower() == github_username_lower:
            return student_id, data
    return None


def update_review_count(
    students: Dict[str, Any],
    reviewer_username: str,
    pr_author_username: str,
    review_state: str,
    pr_number: int
) -> bool:
    """
    Update the review count for a student.

    Args:
        students: The students data dictionary
        reviewer_username: GitHub username of the reviewer
        pr_author_username: GitHub username of the PR author
        review_state: State of the review (approved, changes_requested, commented)
        pr_number: Pull request number

    Returns:
        True if update was successful, False otherwise
    """
    # Don't count self-reviews
    if reviewer_username.lower() == pr_author_username.lower():
        print(f"Skipping self-review by {reviewer_username}")
        return False

    # Find the reviewer in the students list
    student_record = find_student_by_github_username(students, reviewer_username)
    if not student_record:
        print(f"Reviewer {reviewer_username} not found in students list")
        return False

    student_id, student_data = student_record

    # Initialize nb_review if it doesn't exist
    if 'nb_review' not in student_data:
        student_data['nb_review'] = 0

    # Initialize review history if tracking detailed reviews (optional)
    if 'review_history' not in student_data:
        student_data['review_history'] = []

    # Increment review count
    old_count = student_data['nb_review']
    student_data['nb_review'] += 1

    # Update last review timestamp
    student_data['last_review_at'] = datetime.utcnow().isoformat() + 'Z'

    # Optionally track review history (can be disabled if not needed)
    review_record = {
        'pr_number': pr_number,
        'pr_author': pr_author_username,
        'review_state': review_state,
        'reviewed_at': student_data['last_review_at']
    }

    # Keep only last 10 reviews to avoid data bloat
    student_data['review_history'].append(review_record)
    student_data['review_history'] = student_data['review_history'][-10:]

    print(f"Updated {reviewer_username} (ID: {student_id}) review count from {old_count} to {student_data['nb_review']}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Update student review count in S3')
    parser.add_argument('--reviewer', required=True, help='GitHub username of the reviewer')
    parser.add_argument('--pr-author', required=True, help='GitHub username of the PR author')
    parser.add_argument('--review-state', required=True, help='Review state (approved, changes_requested, commented)')
    parser.add_argument('--pr-number', type=int, required=True, help='Pull request number')
    parser.add_argument('--bucket', default='www.raphaelcousin.com', help='S3 bucket name')
    parser.add_argument('--repo-name', required=True, help='Repository name')
    parser.add_argument('--year', default='2025', help='Course year')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run without updating S3')

    args = parser.parse_args()

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Construct S3 key
    s3_key = f"repositories/{args.repo_name}/students/config/students.json"

    print(f"Processing review by {args.reviewer} on PR #{args.pr_number} by {args.pr_author}")
    print(f"Review state: {args.review_state}")

    # Download current students data
    print(f"Downloading students.json from s3://{args.bucket}/{s3_key}")
    students = download_students_json(s3_client, args.bucket, s3_key)

    # Update review count
    if update_review_count(students, args.reviewer, args.pr_author, args.review_state, args.pr_number):
        if not args.dry_run:
            # Upload updated data back to S3
            upload_students_json(s3_client, args.bucket, s3_key, students)
            print("Review count updated successfully")
        else:
            print("Dry run - no changes uploaded to S3")
            print("\nUpdated students.json preview:")
            print(json.dumps(students, indent=2))
    else:
        print("No update performed")
        sys.exit(0)


if __name__ == "__main__":
    main()