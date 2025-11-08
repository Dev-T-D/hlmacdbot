# GitHub Actions CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing, linting, and code quality checks.

## Workflows

### 1. Main CI Pipeline (`ci.yml`)

**Purpose**: Run comprehensive tests and checks on every push and pull request.

**Jobs**:
- **Test**: Runs unit and integration tests on Python 3.11 and 3.12
- **Lint**: Checks code formatting, linting, import sorting, and type checking
- **Security**: Scans for private keys and hardcoded credentials
- **Build**: Verifies all module imports work correctly

**Triggers**:
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches

**Matrix Strategy**:
- Python versions: 3.11, 3.12
- OS: Ubuntu latest

### 2. Code Quality (`code-quality.yml`)

**Purpose**: Comprehensive code quality checks including formatting, linting, and type checking.

**Checks**:
- Black formatter (check mode)
- Flake8 linter with complexity analysis
- isort import sorting (check mode)
- mypy type checking
- TODO/FIXME comment detection (informational)

**Triggers**:
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches
- Weekly schedule (Sundays at 00:00 UTC)

### 3. Test Coverage (`test-coverage.yml`)

**Purpose**: Generate and report test coverage metrics.

**Features**:
- Runs all tests with coverage tracking
- Generates coverage report and XML
- Optional Codecov integration (if configured)

**Triggers**:
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches

## Usage

### Running Locally

To run the same checks locally before pushing:

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests
python run_tests.py

# Check formatting
black --check --line-length=100 --exclude="venv|\.venv|test_.*\.py" .

# Run linter
flake8 --max-line-length=100 --extend-ignore=E203,W503 --exclude=venv,.venv,__pycache__,.git .

# Check imports
isort --check-only --profile=black --line-length=100 --skip="venv|\.venv" .

# Type check
mypy --ignore-missing-imports --no-strict-optional --warn-return-any --exclude="test_.*\.py|venv|\.venv" .
```

### Pre-commit Hooks

For automatic checks before each commit, set up pre-commit hooks:

```bash
./setup_pre_commit.sh
```

This will run the same checks automatically on `git commit`.

## Configuration

### Environment Variables

The workflows use the following environment variables (set in GitHub Secrets if needed):

- `HYPERLIQUID_TESTNET`: Set to `'true'` for testnet testing (default in CI)

### Secrets

No secrets are required for the CI workflows as tests use mocked API responses.

## Workflow Status

You can view workflow status:
- In the GitHub Actions tab of your repository
- Via status badges (add to README.md if desired)

## Troubleshooting

### Tests Failing Locally but Passing in CI

1. Ensure you're using the same Python version (3.11 or 3.12)
2. Install all dependencies: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
3. Clear Python cache: `find . -type d -name __pycache__ -exec rm -r {} +`

### Linting Failures

1. Run `black` to auto-format: `black --line-length=100 --exclude="venv|\.venv|test_.*\.py" .`
2. Run `isort` to fix imports: `isort --profile=black --line-length=100 --skip="venv|\.venv" .`
3. Fix Flake8 errors manually

### Type Checking Failures

1. Review mypy errors and add type hints where needed
2. Use `# type: ignore` comments sparingly and only when necessary
3. Ensure all imports are available (some may be optional dependencies)

## Adding New Workflows

To add a new workflow:

1. Create a new `.yml` file in `.github/workflows/`
2. Follow the existing workflow structure
3. Test locally before pushing
4. Document in this README

## Best Practices

1. **Always run tests locally** before pushing
2. **Fix linting errors** before committing
3. **Keep workflows fast** - use caching and parallel jobs
4. **Don't skip CI checks** unless absolutely necessary
5. **Review workflow logs** if tests fail unexpectedly

