#!/bin/bash

# Deployment script for data science practice website
# This script:
# 1. Builds the React website using npm run build
# 2. Uploads files to S3 bucket www.raphaelcousin.com
# 3. Creates CloudFront invalidation for distribution E15896343L2RDG

set -e  # Exit on any error

# Configuration
S3_BUCKET="www.raphaelcousin.com"
CLOUDFRONT_DISTRIBUTION_ID="E15896343L2RDG"
BUILD_DIR="website/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting deployment process...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    echo "Install with: brew install awscli"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

# Step 1: Build the website
echo -e "${YELLOW}📦 Building website...${NC}"
cd website
npm run build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

cd ..
echo -e "${GREEN}✅ Build completed successfully${NC}"

# Step 2: Upload to S3
echo -e "${YELLOW}☁️  Uploading to S3...${NC}"

# Sync build directory to S3 bucket
aws s3 sync $BUILD_DIR s3://$S3_BUCKET --delete --region eu-west-3

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ S3 upload failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Files uploaded to S3 successfully${NC}"

# Step 3: Create CloudFront invalidation
echo -e "${YELLOW}🌐 Creating CloudFront invalidation...${NC}"

# Create invalidation for all files
INVALIDATION_ID=$(aws cloudfront create-invalidation \
    --distribution-id $CLOUDFRONT_DISTRIBUTION_ID \
    --paths "/*" \
    --region us-east-1 \
    --query 'Invalidation.Id' \
    --output text)

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ CloudFront invalidation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ CloudFront invalidation created with ID: $INVALIDATION_ID${NC}"
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${YELLOW}📊 You can monitor the invalidation status at:${NC}"
echo "https://us-east-1.console.aws.amazon.com/cloudfront/v4/home?region=eu-west-3#/distributions/$CLOUDFRONT_DISTRIBUTION_ID/invalidations"

# Optional: Wait for invalidation to complete
read -p "Do you want to wait for invalidation to complete? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⏳ Waiting for invalidation to complete...${NC}"
    aws cloudfront wait invalidation-completed \
        --distribution-id $CLOUDFRONT_DISTRIBUTION_ID \
        --id $INVALIDATION_ID \
        --region us-east-1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Invalidation completed!${NC}"
        echo -e "${GREEN}🌐 Website is now live at: https://$S3_BUCKET${NC}"
    else
        echo -e "${YELLOW}⚠️  Invalidation is still in progress. Check the console for status.${NC}"
    fi
fi