# Testing Guide

This project uses **Google Test** for modern, maintainable unit testing.

## Project Structure

```
graph-cut-ransac/
├── tests/              # Unit tests
│   └── unit_tests.cpp  # Google Test-based unit tests
├── examples/           # Example applications and notebooks
├── src/                # Library source code
└── build/              # Build output (generated)
    └── unit_tests      # Compiled test executable
```

## Quick Start

### Build and Run Tests

```bash
# Configure with CMake (Google Test will be automatically downloaded)
cmake -B build -DUNIT_TESTS=ON

# Build the project and tests
cmake --build build

# Run all tests with CTest (recommended)
cd build
ctest --output-on-failure

# Or run the test executable directly
./build/unit_tests

# Run tests with colored output and more details
./build/unit_tests --gtest_color=yes
```

### Run Specific Tests

```bash
# Run only tests matching a pattern
./build/unit_tests --gtest_filter=MathUtils.*

# Run tests from a specific test suite
./build/unit_tests --gtest_filter=ConvexHull.*

# Run a specific test
./build/unit_tests --gtest_filter=AngleUtils.ClipAngle

# List all available tests
./build/unit_tests --gtest_list_tests
```

### Test Output Options

```bash
# Verbose output
./build/unit_tests --gtest_verbose

# Repeat tests (useful for flaky tests)
./build/unit_tests --gtest_repeat=10

# Run tests in random order
./build/unit_tests --gtest_shuffle

# Generate XML report (for CI/CD)
./build/unit_tests --gtest_output=xml:test_results.xml
```

## Test Structure

The tests are organized into test suites using Google Test:

- **MathUtils**: Tests for mathematical utility functions
- **ConvexHull**: Tests for convex hull computation
- **PointInPolygonTest**: Tests for point-in-polygon checks (uses test fixtures)
- **LineUtils**: Tests for line geometry functions
- **AngleUtils**: Tests for angle manipulation
- **Combinatorics**: Tests for combinatorial functions
- **RectifyingHomography**: Tests for homography rectification

## Writing New Tests

### Simple Test

```cpp
TEST(TestSuiteName, TestName) {
    EXPECT_EQ(actual, expected);
    EXPECT_NEAR(actual, expected, tolerance);
    EXPECT_TRUE(condition);
    ASSERT_EQ(actual, expected);  // Stops test on failure
}
```

### Test with Fixture (Shared Setup)

```cpp
class MyTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code runs before each test
    }
    
    void TearDown() override {
        // Cleanup code runs after each test
    }
    
    // Shared data
    int shared_value;
};

TEST_F(MyTestFixture, TestName) {
    EXPECT_EQ(shared_value, expected);
}
```

### Parameterized Tests

```cpp
class MyParamTest : public ::testing::TestWithParam<int> {};

TEST_P(MyParamTest, TestName) {
    int param = GetParam();
    EXPECT_GT(param, 0);
}

INSTANTIATE_TEST_SUITE_P(
    MyTestSuite,
    MyParamTest,
    ::testing::Values(1, 2, 3, 4, 5)
);
```

## Benefits of Google Test

Google Test provides many advantages over custom test frameworks:

| Feature | Benefit |
|---------|---------|
| Test discovery | Automatically finds and registers all tests |
| Run specific tests | Filter tests by name or pattern |
| Test fixtures | Built-in support for shared test setup/teardown |
| Parameterized tests | Run same test with different inputs |
| XML output (CI/CD) | Generate reports for continuous integration |
| Rich assertions | Extensive assertion macros with clear error messages |
| Filter tests | Run subset of tests easily |
| Colored output | Color-coded test results |
| Test execution order | Configurable (sequential, random, etc.) |
| Integration with IDEs | Native support in CLion, VS Code, Visual Studio |

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Build and Test
  run: |
    cmake -B build -DUNIT_TESTS=ON
    cmake --build build
    cd build && ctest --output-on-failure
```

### GitLab CI Example

```yaml
test:
  script:
    - cmake -B build -DUNIT_TESTS=ON
    - cmake --build build
    - cd build && ctest --output-on-failure --verbose
```

## IDE Integration

Most modern IDEs (VS Code, CLion, Visual Studio) can detect Google Test tests and provide:
- Test explorer UI
- Run/debug individual tests
- Test results inline in the editor

### VS Code

Install the "C++ TestMate" extension for test discovery and execution.

### CLion

Google Test is natively supported - tests appear in the test runner automatically.

## References

- [Google Test Documentation](https://google.github.io/googletest/)
- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [Google Test Advanced Guide](https://google.github.io/googletest/advanced.html)

