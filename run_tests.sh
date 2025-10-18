#!/bin/bash

# Script to easily build and run tests
# Usage: ./run_tests.sh [options]

set -e  # Exit on error

# Default values
BUILD_DIR="build"
CLEAN_BUILD=false
VERBOSE=false
FILTER=""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Help message
show_help() {
    cat << EOF
Usage: ./run_tests.sh [OPTIONS]

Build and run unit tests for Graph-Cut RANSAC using Google Test.

OPTIONS:
    -h, --help              Show this help message
    -c, --clean             Clean build directory before building
    -v, --verbose           Show verbose build output
    -f, --filter PATTERN    Run only tests matching PATTERN
    --list                  List all available tests

EXAMPLES:
    ./run_tests.sh                          # Build and run all tests
    ./run_tests.sh -c                       # Clean build and run all tests
    ./run_tests.sh -f "MathUtils.*"         # Run only MathUtils tests
    ./run_tests.sh --list                   # List all tests

EOF
}

# Parse command line arguments
LIST_TESTS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--filter)
            FILTER="$2"
            shift 2
            ;;
        --list)
            LIST_TESTS=true
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${BLUE}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Configure with CMake
echo -e "${BLUE}Configuring CMake...${NC}"
CMAKE_OPTS="-B $BUILD_DIR -DUNIT_TESTS=ON"

if [ "$VERBOSE" = true ]; then
    cmake $CMAKE_OPTS
else
    cmake $CMAKE_OPTS > /dev/null
fi

# Build
echo -e "${BLUE}Building tests...${NC}"
if [ "$VERBOSE" = true ]; then
    cmake --build "$BUILD_DIR"
else
    cmake --build "$BUILD_DIR" 2>&1 | grep -E "(error|warning|Building|Linking)" || true
fi

echo -e "${GREEN}Build complete!${NC}\n"

# Run tests
if [ "$LIST_TESTS" = true ]; then
    echo -e "${BLUE}Available tests:${NC}"
    ./"$BUILD_DIR"/unit_tests --gtest_list_tests
else
    echo -e "${BLUE}Running tests...${NC}\n"
    if [ -n "$FILTER" ]; then
        ./"$BUILD_DIR"/unit_tests --gtest_color=yes --gtest_filter="$FILTER"
    else
        ./"$BUILD_DIR"/unit_tests --gtest_color=yes
    fi
fi

echo -e "\n${GREEN}All tests completed!${NC}"

