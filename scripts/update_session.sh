#!/bin/bash

# Define the source directory (assumed to be one level up)
SOURCE_DIR="../data_science_practice"

# Define the list of items to copy
ITEMS_TO_COPY=(
  "scripts"
  "tests"
  ".github"
  ".gitignore"
  "README.md"
)

# Loop through each item and copy it to the current directory
for item in "${ITEMS_TO_COPY[@]}"; do
  if [ -e "$SOURCE_DIR/$item" ]; then
    echo "Copying $item from $SOURCE_DIR to current directory..."
    cp -R "$SOURCE_DIR/$item" .
  else
    echo "Warning: $item does not exist in $SOURCE_DIR."
  fi
done

echo "All specified files and folders have been copied/replaced."
