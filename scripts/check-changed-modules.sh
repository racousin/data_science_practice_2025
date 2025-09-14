#!/bin/bash
# Script to determine changed modules and optionally fetch all modules

USER=$1
ALL_MODULES=${2:-false}  # Default to false if not provided

if [ "$ALL_MODULES" = true ]; then
    # List all modules for the user
    git ls-files | grep "^${USER}/module" | sed -E 's|.*/module([0-9]+)/.*|\1|' | sort -u | tr '\n' ' ' | xargs
else
    # List only changed modules
    git diff --name-only HEAD^ HEAD | grep "^${USER}/module" | sed -E 's|.*/module([0-9]+)/.*|\1|' | sort -u | tr '\n' ' ' | xargs
fi
