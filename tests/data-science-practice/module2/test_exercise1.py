# mysupertools/tests/test_multiplication.py

import os
import glob
from mysupertools.tool.operation_a_b import multiply


def test_multiply_numbers():
    assert multiply(4, 5) == 20
    assert multiply(-1, 5) == -5


def test_multiply_errors():
    assert multiply("a", 5) == "error"
    assert multiply(None, 5) == "error"


def test_no_build_files_pushed():
    """Test that build and egg folder files are not present in the repository"""
    # Get the package directory (should be one level up from where this test is running)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for the mysupertools package in the student directory structure
    # This assumes the test is running from the correct location
    package_search_paths = [
        os.path.join(current_dir, "..", "..", "..", "*", "module2", "mysupertools"),
        "mysupertools"  # fallback if running from package directory
    ]

    package_dir = None
    for search_path in package_search_paths:
        found_dirs = glob.glob(search_path)
        if found_dirs:
            package_dir = found_dirs[0]
            break

    if not package_dir or not os.path.exists(package_dir):
        # If we can't find the package directory, skip this test
        return

    # Check for common build artifacts that should not be committed
    build_patterns = [
        "build/*",
        "dist/*",
        "*.egg-info/*",
        "*.egg/*",
        "__pycache__/*",
        "*.pyc",
        "*.pyo"
    ]

    found_artifacts = []
    for pattern in build_patterns:
        search_pattern = os.path.join(package_dir, "**", pattern)
        artifacts = glob.glob(search_pattern, recursive=True)
        found_artifacts.extend(artifacts)

    # Also check for these directories at the package root
    root_build_dirs = ["build", "dist"]
    for dirname in root_build_dirs:
        full_path = os.path.join(package_dir, dirname)
        if os.path.exists(full_path):
            found_artifacts.append(full_path)

    # Check for .egg-info directories
    egg_info_dirs = glob.glob(os.path.join(package_dir, "*.egg-info"))
    found_artifacts.extend(egg_info_dirs)

    assert len(found_artifacts) == 0, f"Build artifacts found that should not be committed: {found_artifacts}"
