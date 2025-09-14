#!/usr/bin/env python3
"""
Generate review statistics from students.json file.
Can be used to track review participation and identify top reviewers.
"""

import json
import sys
import argparse
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple


def download_students_json(s3_client, bucket: str, key: str) -> Dict[str, Any]:
    """Download students.json from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        print(f"Error downloading students.json: {e}")
        sys.exit(1)


def calculate_review_statistics(students: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate various review statistics."""
    stats = {
        'total_reviews': 0,
        'students_with_reviews': 0,
        'students_without_reviews': 0,
        'average_reviews_per_student': 0,
        'top_reviewers': [],
        'recent_reviewers': [],
        'review_distribution': {}
    }

    review_counts = []
    students_with_reviews = []
    students_without_reviews = []

    for student_id, data in students.items():
        nb_review = data.get('nb_review', 0)
        github_username = data.get('github_username', student_id)
        firstname = data.get('firstname', '')
        lastname = data.get('lastname', '')
        full_name = f"{firstname} {lastname}".strip() or github_username

        review_counts.append(nb_review)
        stats['total_reviews'] += nb_review

        if nb_review > 0:
            students_with_reviews.append({
                'student_id': student_id,
                'name': full_name,
                'github_username': github_username,
                'nb_review': nb_review,
                'last_review_at': data.get('last_review_at', 'N/A')
            })
        else:
            students_without_reviews.append({
                'student_id': student_id,
                'name': full_name,
                'github_username': github_username
            })

        # Track distribution
        if nb_review not in stats['review_distribution']:
            stats['review_distribution'][nb_review] = 0
        stats['review_distribution'][nb_review] += 1

    stats['students_with_reviews'] = len(students_with_reviews)
    stats['students_without_reviews'] = len(students_without_reviews)

    if review_counts:
        stats['average_reviews_per_student'] = sum(review_counts) / len(review_counts)

    # Get top reviewers
    students_with_reviews.sort(key=lambda x: x['nb_review'], reverse=True)
    stats['top_reviewers'] = students_with_reviews[:10]

    # Get recent reviewers (last 7 days)
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)

    for student in students_with_reviews:
        try:
            last_review = datetime.fromisoformat(student['last_review_at'].replace('Z', '+00:00'))
            if last_review.replace(tzinfo=None) > week_ago:
                stats['recent_reviewers'].append(student)
        except:
            pass

    stats['recent_reviewers'].sort(key=lambda x: x['last_review_at'], reverse=True)

    return stats, students_without_reviews


def print_statistics(stats: Dict[str, Any], students_without_reviews: List[Dict[str, Any]], verbose: bool = False):
    """Print formatted statistics."""
    print("\n" + "="*60)
    print("📊 REVIEW STATISTICS")
    print("="*60)

    print(f"\n📈 Summary:")
    print(f"  • Total reviews completed: {stats['total_reviews']}")
    print(f"  • Students who have reviewed: {stats['students_with_reviews']}")
    print(f"  • Students who haven't reviewed: {stats['students_without_reviews']}")
    print(f"  • Average reviews per student: {stats['average_reviews_per_student']:.2f}")

    print(f"\n🏆 Top 10 Reviewers:")
    for i, reviewer in enumerate(stats['top_reviewers'], 1):
        print(f"  {i:2}. {reviewer['name']:30} (@{reviewer['github_username']:20}) - {reviewer['nb_review']} reviews")

    if stats['recent_reviewers']:
        print(f"\n⏰ Recent Reviewers (last 7 days):")
        for reviewer in stats['recent_reviewers'][:5]:
            print(f"  • {reviewer['name']:30} (@{reviewer['github_username']:20}) - Last: {reviewer['last_review_at'][:10]}")

    print(f"\n📊 Review Distribution:")
    for count in sorted(stats['review_distribution'].keys()):
        num_students = stats['review_distribution'][count]
        bar = '█' * min(num_students, 40)
        print(f"  {count:2} reviews: {bar} ({num_students} students)")

    if verbose and students_without_reviews:
        print(f"\n⚠️  Students Without Reviews:")
        for student in students_without_reviews:
            print(f"  • {student['name']:30} (@{student['github_username']})")

    print("\n" + "="*60)


def export_to_csv(stats: Dict[str, Any], students: Dict[str, Any], output_file: str):
    """Export review data to CSV format."""
    import csv

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['student_id', 'firstname', 'lastname', 'github_username', 'nb_review', 'last_review_at', 'progress_percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for student_id, data in students.items():
            writer.writerow({
                'student_id': student_id,
                'firstname': data.get('firstname', ''),
                'lastname': data.get('lastname', ''),
                'github_username': data.get('github_username', ''),
                'nb_review': data.get('nb_review', 0),
                'last_review_at': data.get('last_review_at', ''),
                'progress_percentage': data.get('progress_percentage', 0)
            })

    print(f"\n📁 Data exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate review statistics from students.json')
    parser.add_argument('--bucket', default='www.raphaelcousin.com', help='S3 bucket name')
    parser.add_argument('--repo-name', required=True, help='Repository name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output including non-reviewers')
    parser.add_argument('--export-csv', help='Export data to CSV file')
    parser.add_argument('--local-file', help='Use local students.json file instead of S3')

    args = parser.parse_args()

    if args.local_file:
        # Load from local file
        print(f"Loading students data from local file: {args.local_file}")
        with open(args.local_file, 'r') as f:
            students = json.load(f)
    else:
        # Load from S3
        s3_client = boto3.client('s3')
        s3_key = f"repositories/{args.repo_name}/students/config/students.json"
        print(f"Loading students data from s3://{args.bucket}/{s3_key}")
        students = download_students_json(s3_client, args.bucket, s3_key)

    # Calculate statistics
    stats, students_without_reviews = calculate_review_statistics(students)

    # Print statistics
    print_statistics(stats, students_without_reviews, args.verbose)

    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(stats, students, args.export_csv)


if __name__ == "__main__":
    main()