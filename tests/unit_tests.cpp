#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "math_utils.hpp"
#include "model.h"

using namespace gcransac::utils;

// ==============================================================================
// Collinear Points Tests
// ==============================================================================

TEST(MathUtils, CollinearPointsPositiveCase) {
    Point2D p0{1, 0};
    Point2D p1{2, 1};
    Point2D p2{3, 2};
    EXPECT_TRUE(areCollinear(p0, p1, p2, 1e-9))
        << "Points (" << p0.x() << ", " << p0.y() << "), "
        << "(" << p1.x() << ", " << p1.y() << "), "
        << "(" << p2.x() << ", " << p2.y() << ") should be collinear";
}

TEST(MathUtils, CollinearPointsNegativeCase) {
    Point2D p0{1, 0};
    Point2D p1{2, 1};
    Point2D p2{1, 4};
    EXPECT_FALSE(areCollinear(p0, p1, p2, 1e-9))
        << "Points (" << p0.x() << ", " << p0.y() << "), "
        << "(" << p1.x() << ", " << p1.y() << "), "
        << "(" << p2.x() << ", " << p2.y() << ") should not be collinear";
}

// ==============================================================================
// Convex Hull Tests
// ==============================================================================

TEST(ConvexHull, StandardCase) {
    std::vector<Point2D> points;
    points.emplace_back(0, 3);
    points.emplace_back(2, 2);
    points.emplace_back(1, 1);
    points.emplace_back(2, 1);
    points.emplace_back(3, 0);
    points.emplace_back(0, 0);
    points.emplace_back(3, 3);
    
    std::vector<Point2D> expected;
    expected.emplace_back(0, 0);
    expected.emplace_back(3, 0);
    expected.emplace_back(3, 3);
    expected.emplace_back(0, 3);
    
    std::vector<Point2D> actual = computeConvexHull(points);
    
    ASSERT_EQ(expected.size(), actual.size())
        << "Convex hull size mismatch";
    
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    
    for (size_t i = 0; i < actual.size(); i++) {
        EXPECT_EQ(actual[i].x(), expected[i].x())
            << "Mismatch at index " << i << " for x coordinate";
        EXPECT_EQ(actual[i].y(), expected[i].y())
            << "Mismatch at index " << i << " for y coordinate";
    }
}

TEST(ConvexHull, Degenerate1D) {
    const Point2D expected_vertex{1.45, -5.2};
    std::vector<Point2D> points{10, expected_vertex};
    
    std::vector<Point2D> actual = computeConvexHull(points);
    
    ASSERT_EQ(1, actual.size())
        << "Expected single vertex for degenerate case";
    
    EXPECT_NEAR(actual[0].x(), expected_vertex.x(), 1e-9);
    EXPECT_NEAR(actual[0].y(), expected_vertex.y(), 1e-9);
}

TEST(ConvexHull, Degenerate2D) {
    const Point2D expected_vertex_1{1.45, -5.2};
    const Point2D expected_vertex_2{-3.14, -1.73};
    std::vector<Point2D> expected{expected_vertex_1, expected_vertex_2};
    
    std::vector<Point2D> points;
    for (size_t i = 0; i < 5; i++) {
        points.push_back(expected_vertex_1);
        points.push_back(expected_vertex_2);
    }
    
    std::vector<Point2D> actual = computeConvexHull(points);
    
    ASSERT_EQ(expected.size(), actual.size())
        << "Convex hull size mismatch for degenerate 2D case";
    
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    
    for (size_t i = 0; i < actual.size(); i++) {
        EXPECT_EQ(actual[i].x(), expected[i].x())
            << "Mismatch at index " << i;
        EXPECT_EQ(actual[i].y(), expected[i].y())
            << "Mismatch at index " << i;
    }
}

// ==============================================================================
// Point in Polygon Tests
// ==============================================================================

class PointInPolygonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a square polygon
        convex_polygon.emplace_back(0, 0);
        convex_polygon.emplace_back(3, 0);
        convex_polygon.emplace_back(3, 3);
        convex_polygon.emplace_back(0, 3);
    }
    
    std::vector<Point2D> convex_polygon;
};

TEST_F(PointInPolygonTest, PointInside) {
    const Point2D point{1, 2};
    EXPECT_TRUE(pointInConvexPolygon(point, convex_polygon))
        << "Point (1, 2) should be inside the square";
}

TEST_F(PointInPolygonTest, PointOutside) {
    const Point2D point{-1, 2};
    EXPECT_FALSE(pointInConvexPolygon(point, convex_polygon))
        << "Point (-1, 2) should be outside the square";
}

TEST_F(PointInPolygonTest, PointOnEdge) {
    const Point2D point{1.5, 1.5};
    EXPECT_TRUE(pointInConvexPolygon(point, convex_polygon))
        << "Point on edge should be considered inside";
}

TEST_F(PointInPolygonTest, PointOnVertex) {
    const Point2D point{3, 3};
    EXPECT_TRUE(pointInConvexPolygon(point, convex_polygon))
        << "Point on vertex should be considered inside";
}

// ==============================================================================
// Line and Angle Tests
// ==============================================================================

TEST(LineUtils, LineFromPointAndAngle) {
    auto line = lineFromPointAndAngle(0.0, 0.0, 0.0);
    
    // Test point on line
    Eigen::Vector3d point{1.0, 0.0, 1.0};
    double dist = std::abs(line.dot(point));
    EXPECT_NEAR(dist, 0.0, 1e-9)
        << "Point (1, 0) should be on line through (0,0) with angle 0";
    
    // Test point off line
    point = {-1.0, 1.0, 1.0};
    dist = std::abs(line.dot(point));
    EXPECT_NEAR(dist, 1.0, 1e-9)
        << "Point (-1, 1) should be 1 unit from line";
    
    // Test intersection
    auto other_line = lineFromPointAndAngle(1.0, 1.0, M_PI_2);
    auto intersection = line.cross(other_line);
    EXPECT_GT(std::abs(intersection(2)), 1e-9)
        << "Lines should intersect at a finite point";
    
    intersection /= intersection(2);
    EXPECT_NEAR(intersection(0), 1.0, 1e-9);
    EXPECT_NEAR(intersection(1), 0.0, 1e-9);
}

TEST(AngleUtils, DegreesToRadians) {
    constexpr double kEpsilon = 1e-9;
    constexpr double kDegs = 90.0;
    constexpr double kRads = M_PI_2;
    
    EXPECT_NEAR(deg2rad(kDegs), kRads, kEpsilon);
    EXPECT_NEAR(rad2deg(kRads), kDegs, kEpsilon);
    EXPECT_NEAR(rad2deg(deg2rad(kDegs)), kDegs, kEpsilon);
}

TEST(AngleUtils, ClipAngle) {
    const std::vector<std::pair<double, double>> test_cases = {
        {-M_PI_2, 1.5 * M_PI},
        {0.7861, 0.7861},
        {0.0, 0.0},
        {2.7 * M_PI, 0.7 * M_PI},
        {-M_PI, M_PI}
    };
    
    for (const auto& [input, expected] : test_cases) {
        EXPECT_NEAR(clipAngle(input), expected, 1e-9)
            << "clipAngle(" << input << ") failed";
    }
}

TEST(AngleUtils, MinAngleDiff) {
    const std::vector<std::tuple<double, double, double>> test_cases = {
        {0.0, 0.0, 0.0},
        {-M_PI_2, 0.0, M_PI_2},
        {-3.48 * M_PI, 7.41 * M_PI, 0.89 * M_PI},
        {0.25 * M_PI, 1.25 * M_PI, M_PI}
    };
    
    for (const auto& [angle1, angle2, expected] : test_cases) {
        EXPECT_NEAR(minAngleDiff(angle1, angle2), expected, 1e-9)
            << "minAngleDiff(" << angle1 << ", " << angle2 << ") failed";
    }
}

TEST(AngleUtils, LinesAnglesDiff) {
    const std::vector<std::tuple<double, double, double>> test_cases = {
        {0.0, 0.0, 0.0},
        {-M_PI_2, 0.0, M_PI_2},
        {0.52 * M_PI, 1.49 * M_PI, 0.03 * M_PI},
        {1.49 * M_PI, 0.52 * M_PI, 0.03 * M_PI},
        {-3.48 * M_PI, 5.60 * M_PI, 0.08 * M_PI},
        {0.25 * M_PI, 1.25 * M_PI, 0.0}
    };
    
    for (const auto& [angle1, angle2, expected] : test_cases) {
        EXPECT_NEAR(linesAnglesDiff(angle1, angle2), expected, 1e-9)
            << "linesAnglesDiff(" << angle1 << ", " << angle2 << ") failed";
    }
}

// ==============================================================================
// Combinatorics Tests
// ==============================================================================

TEST(Combinatorics, NChoose2) {
    EXPECT_EQ(nChoose2(6), 15);
    EXPECT_EQ(nChoose2(1), 0);
    EXPECT_EQ(nChoose2(0), 0);
}

// ==============================================================================
// Rectifying Homography Tests
// ==============================================================================

TEST(RectifyingHomography, WarpingAndRectifyingConsistency) {
    constexpr double kEpsilon = 1e-12;
    
    gcransac::RectifyingHomography model;
    model.h7 = 0.0001;
    model.h8 = 0.0002;

    const double udx{82.4};
    const double udy{-12.3};
    const double uds{1.13};
    const double udt{0.56};
    
    // Warp x, y, scale, and angle
    auto ds = model.unrectifiedScale(udx, udy, uds);
    auto dt = model.unrectifiedAngle(udx, udy, udt);
    Eigen::Vector3d p(udx, udy, 1.0);
    model.unrectifyPoint(p);
    auto dx = p(0) / p(2);
    auto dy = p(1) / p(2);
    
    // Rectify x, y, scale, and angle
    auto uds_comp = model.rectifiedScale(dx, dy, ds);
    auto udt_comp = model.rectifiedAngle(dx, dy, dt);
    p = Eigen::Vector3d(dx, dy, 1.0);
    model.rectifyPoint(p);
    auto udx_comp = p(0) / p(2);
    auto udy_comp = p(1) / p(2);
    
    // Verify consistency
    EXPECT_NEAR(udx, udx_comp, kEpsilon) << "X coordinate mismatch";
    EXPECT_NEAR(udy, udy_comp, kEpsilon) << "Y coordinate mismatch";
    EXPECT_NEAR(uds, uds_comp, kEpsilon) << "Scale mismatch";
    EXPECT_NEAR(udt, udt_comp, kEpsilon) << "Angle mismatch";
}

// ==============================================================================
// Main function
// ==============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

