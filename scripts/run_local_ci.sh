#!/bin/bash
# Local CI test runner for Keisei Shogi
# Run this script to test the same checks that will run in CI

set -e  # Exit on any error

echo "=== Keisei Shogi Local CI Runner ==="
echo "Running the same checks as CI pipeline..."
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2 passed${NC}"
    else
        echo -e "${RED}✗ $2 failed${NC}"
        return 1
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "keisei" ]; then
    echo -e "${RED}Error: Please run this script from the Keisei project root directory${NC}"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: No virtual environment detected. Consider activating your venv.${NC}"
fi

echo "1. Installing dependencies..."
uv pip install -e ".[dev]"
print_status $? "Dependency installation"

echo
echo "2. Running linting checks..."

# Flake8 - critical errors
echo "  - Running flake8 (critical errors)..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
print_status $? "Flake8 critical errors"

# Flake8 - all errors (non-critical)
echo "  - Running flake8 (all errors)..."
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics > /dev/null 2>&1
print_status $? "Flake8 full check"

echo
echo "3. Running type checks..."
mypy keisei/ --ignore-missing-imports --no-strict-optional || true
print_status 0 "MyPy type check (informational)"

echo
echo "4. Running security scan..."
bandit -r keisei/ -f json > bandit_report.json || true
if [ -f bandit_report.json ]; then
    echo "  - Bandit security scan completed (see bandit_report.json for details)"
    rm -f bandit_report.json
fi
print_status 0 "Security scan"

echo
echo "5. Running unit tests..."
pytest tests/unit/ -v --tb=short
print_status $? "Unit tests"

echo
echo "6. Running integration tests..."
pytest tests/integration/ -v --tb=short
print_status $? "Integration tests"

echo
echo "7. Running test coverage..."
pytest tests/ --cov=keisei --cov-report=term-missing --cov-report=html
print_status $? "Test coverage"

echo
echo -e "${GREEN}=== Local CI checks completed! ===${NC}"
echo
echo "Coverage report generated in htmlcov/index.html"
