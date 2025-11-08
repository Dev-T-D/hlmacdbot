#!/bin/bash
# Setup script for pre-commit hooks
# Run this script to install pre-commit hooks for code quality checks

set -e

echo "=========================================="
echo "Pre-commit Hooks Setup"
echo "=========================================="
echo ""

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
else
    echo "✅ Pre-commit is already installed"
fi

# Install development dependencies
echo ""
echo "Installing development dependencies..."
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
else
    echo "⚠️  requirements-dev.txt not found, installing minimal dependencies..."
    pip install pre-commit black flake8 mypy isort
fi

# Install pre-commit hooks
echo ""
echo "Installing pre-commit hooks..."
pre-commit install

echo ""
echo "=========================================="
echo "✅ Pre-commit hooks installed successfully!"
echo "=========================================="
echo ""
echo "Hooks will run automatically on 'git commit'"
echo ""
echo "To run hooks manually on all files:"
echo "  pre-commit run --all-files"
echo ""
echo "To skip hooks for a commit:"
echo "  git commit --no-verify"
echo ""

