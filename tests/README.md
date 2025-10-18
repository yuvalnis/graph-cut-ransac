# Unit Tests

This directory contains unit tests for the Graph-Cut RANSAC library.

## Files

- **`unit_tests.cpp`** - Google Test-based unit tests for core library functionality

## Test Coverage

The unit tests cover:
- Mathematical utility functions (`MathUtils`)
- Convex hull computation (`ConvexHull`)
- Point-in-polygon checks (`PointInPolygonTest`)
- Line geometry functions (`LineUtils`)
- Angle manipulation (`AngleUtils`)
- Combinatorial functions (`Combinatorics`)
- Rectifying homography transformations (`RectifyingHomography`)

## Running Tests

See the main [TESTING.md](../TESTING.md) documentation for details on building and running tests.

### Quick Start

```bash
# From project root
./run_tests.sh

# Or manually
cmake -B build -DUNIT_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

## Adding New Tests

When adding new tests to `unit_tests.cpp`:

1. **Include necessary headers** at the top
2. **Use appropriate test suite names** (e.g., `TEST(SuiteName, TestName)`)
3. **Use test fixtures** for shared setup with `TEST_F(FixtureName, TestName)`
4. **Use descriptive assertions** with clear error messages
5. **Tests are automatically discovered** - no manual registration needed

### Example

```cpp
TEST(MySuite, MyNewTest) {
    // Arrange
    int expected = 42;
    
    // Act
    int actual = myFunction();
    
    // Assert
    EXPECT_EQ(actual, expected) << "myFunction() returned incorrect value";
}
```

## Test Organization

Tests are organized by functionality into logical test suites. When adding new tests:
- Group related tests in the same test suite
- Use descriptive test names that explain what is being tested
- Add section comments for clarity

## Dependencies

- Google Test (automatically downloaded by CMake)
- Eigen3 (linear algebra)
- OpenCV (optional, for some tests)

