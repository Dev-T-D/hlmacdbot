# Pre-commit Hooks Setup Guide

This project uses [pre-commit](https://pre-commit.com/) hooks to ensure code quality and consistency before commits.

## 🎯 What Are Pre-commit Hooks?

Pre-commit hooks automatically run code quality checks before each `git commit`. They help catch:
- Code formatting issues
- Linting errors
- Type checking problems
- Security issues (like accidentally committing private keys)
- Common mistakes (trailing whitespace, merge conflicts, etc.)

## 📦 Installation

### Option 1: Automated Setup (Recommended)

```bash
# Make script executable (if not already)
chmod +x setup_pre_commit.sh

# Run setup script
./setup_pre_commit.sh
```

### Option 2: Manual Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## ✅ What Hooks Are Configured?

### General File Checks
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newline
- **check-yaml**: Validates YAML files
- **check-json**: Validates JSON files
- **check-added-large-files**: Prevents committing large files (>1MB)
- **check-merge-conflict**: Detects merge conflict markers
- **check-case-conflict**: Detects case conflicts in filenames
- **detect-private-key**: Detects accidentally committed private keys
- **mixed-line-ending**: Ensures consistent line endings (LF)

### Code Formatting
- **Black**: Automatically formats Python code (100-character line length)
- **isort**: Sorts imports (compatible with Black)

### Code Quality
- **Flake8**: Lints Python code for style and errors
  - Includes `flake8-docstrings` for docstring checking
  - Includes `flake8-bugbear` for common bug detection
- **MyPy**: Type checks Python code (excludes test files)

## 🚀 Usage

### Automatic (Default)
Hooks run automatically when you commit:
```bash
git add .
git commit -m "Your commit message"
# Hooks run automatically here
```

### Manual Run
Run hooks on all files (useful before pushing):
```bash
pre-commit run --all-files
```

### Run Specific Hook
Run a specific hook:
```bash
pre-commit run black --all-files
pre-commit run flake8 --all-files
pre-commit run mypy --all-files
```

### Skip Hooks (Not Recommended)
Skip hooks for a specific commit:
```bash
git commit --no-verify -m "Emergency fix"
```

⚠️ **Warning**: Only skip hooks when absolutely necessary. Skipping hooks can introduce code quality issues.

## 🔧 Configuration

The pre-commit configuration is in `.pre-commit-config.yaml`. Key settings:

- **Line Length**: 100 characters (matches project style)
- **Exclusions**: Test files (`test_*.py`) and virtual environments are excluded from some checks
- **Black Profile**: isort uses Black-compatible profile

## 📝 Fixing Issues

### Auto-fixable Issues
Many issues are automatically fixed:
- Trailing whitespace
- End-of-file newlines
- Code formatting (Black)
- Import sorting (isort)

### Manual Fixes Required
Some issues require manual fixes:
- Flake8 errors (unused variables, complexity, etc.)
- MyPy type errors
- Security issues (private keys detected)

### Common Fixes

**Black formatting:**
```bash
# Auto-format all files
pre-commit run black --all-files
```

**Import sorting:**
```bash
# Sort imports
pre-commit run isort --all-files
```

**Flake8 errors:**
```bash
# See all linting errors
pre-commit run flake8 --all-files
# Fix manually based on error messages
```

## 🔄 Updating Hooks

Update hooks to latest versions:
```bash
pre-commit autoupdate
```

## 🐛 Troubleshooting

### Hooks Not Running
```bash
# Reinstall hooks
pre-commit install --install-hooks
```

### Hook Fails on Existing Code
```bash
# Run hooks on all files to fix issues
pre-commit run --all-files
```

### MyPy Errors
MyPy is configured to ignore missing imports. If you see type errors:
1. Check if types are available: `pip install types-requests` (for requests)
2. Add type hints to your code
3. Use `# type: ignore` comments for unavoidable issues

### Black Conflicts with Flake8
Black and Flake8 are configured to be compatible:
- Flake8 ignores E203 (whitespace before ':')
- Flake8 ignores W503 (line break before binary operator)

## 📚 Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)

## ✅ Benefits

- **Consistent Code Style**: All code follows the same formatting rules
- **Early Error Detection**: Catch issues before they reach the repository
- **Security**: Prevents accidentally committing secrets
- **Better Code Quality**: Automated checks ensure high standards
- **Team Collaboration**: Everyone follows the same standards

