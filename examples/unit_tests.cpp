#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "math_utils.hpp"

void testCollinearPointsPositiveCase()
{
    using namespace gcransac::utils;
    Point2D p0{1, 0};
    Point2D p1{2, 1};
    Point2D p2{3, 2};
    if (!areCollinear(p0, p1, p2, 1e-9))
    {
        std::stringstream error_msg;
        error_msg << "Incorrect result that points "
                  << "(" << p0.x() << ", " << p0.y() << "), "
                  << "(" << p1.x() << ", " << p1.y() << "), "
                  << "(" << p2.x() << ", " << p2.y() << ") "
                  << "are not collinear.\n";
        throw std::runtime_error(error_msg.str());
    }
}

void testCollinearPointsNegativeCase()
{
    using namespace gcransac::utils;
    Point2D p0{1, 0};
    Point2D p1{2, 1};
    Point2D p2{1, 4};
    if (areCollinear(p0, p1, p2, 1e-9))
    {
        std::stringstream error_msg;
        error_msg << "Incorrect result that points "
                  << "(" << p0.x() << ", " << p0.y() << "), "
                  << "(" << p1.x() << ", " << p1.y() << "), "
                  << "(" << p2.x() << ", " << p2.y() << ") "
                  << "are collinear.\n";
        throw std::runtime_error(error_msg.str());
    }
}

void testConvexHullAlgo()
{
    using namespace gcransac::utils;
    // Construct input
    std::vector<Point2D> points;
    points.emplace_back(0, 3);
    points.emplace_back(2, 2);
    points.emplace_back(1, 1);
    points.emplace_back(2, 1);
    points.emplace_back(3, 0);
    points.emplace_back(0, 0);
    points.emplace_back(3, 3);
    // Constuct expected output
    std::vector<Point2D> expected;
    expected.emplace_back(0, 0);
    expected.emplace_back(3, 0);
    expected.emplace_back(3, 3);
    expected.emplace_back(0, 3);
    // Find the convex hull
    std::vector<Point2D> actual = computeConvexHull(points);
    // Verify expected and actual output are the same size
    if (expected.size() != actual.size())
    {
        std::stringstream error_msg;
        error_msg << "Expected convex-hull size is " << expected.size()
                  << " but actual convex-hull size is " << actual.size()
                  << "\n";
        throw std::runtime_error(error_msg.str());
    }
    // Sort actual and expected outputs
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    // Verify both are equal
    for (size_t i = 0; i < actual.size(); i++)
    {
        const auto& p = actual.at(i);
        const auto& q = expected.at(i);
        if (p.x() != q.x() || p.y() != q.y())
        {
            std::stringstream error_msg;
            error_msg << "Actual convex-hull is different than expected. "
                      << "Point at index " << i
                      << " is (" << p.x() << ", " << p.y() << "), when "
                      << "(" << q.x() << ", " << q.y() << "), was expected.\n";
            throw std::runtime_error(error_msg.str());
        }
    }
}

void testDegenerateConvexHull1D()
{
    using namespace gcransac::utils;
    const Point2D expected_vertex{1.45, -5.2};
    // Construct input
    std::vector<Point2D> points{10, expected_vertex};
    // Find the convex hull
    std::vector<Point2D> actual = computeConvexHull(points);
    // Verify expected and actual output are the same size
    if (actual.size() != 1)
    {
        std::stringstream error_msg;
        error_msg << "Expected convex-hull size is 1, but actual convex-hull "
                     "size is " << actual.size() << std::endl;
        throw std::runtime_error(error_msg.str());
    }
    const auto& actual_vertex = actual.at(0);
    if ((std::abs(actual_vertex.x() - expected_vertex.x()) > 1e-9) ||
        (std::abs(actual_vertex.y() - expected_vertex.y()) > 1e-9))
    {
        std::stringstream error_msg;
        error_msg << "Expected convex-hull single vertex to be "
                  << expected_vertex.toString() << ", but it actually is "
                  << actual_vertex.toString() << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

void testDegenerateConvexHull2D()
{
    using namespace gcransac::utils;
    // Construct expected output
    const Point2D expected_vertex_1{1.45, -5.2};
    const Point2D expected_vertex_2{-3.14, -1.73};
    std::vector<Point2D> expected{expected_vertex_1, expected_vertex_2};
    // Construct input
    std::vector<Point2D> points;
    for (size_t i = 0; i < 5; i++)
    {
        points.push_back(expected_vertex_1);
        points.push_back(expected_vertex_2);
    }
    // Find the convex hull
    std::vector<Point2D> actual = computeConvexHull(points);
    // Verify expected and actual output are the same size
    if (actual.size() != expected.size())
    {
        std::stringstream error_msg;
        error_msg << "Expected convex-hull size is " << expected.size()
                  << ", but actual convex-hull size is " << actual.size()
                  << std::endl;
        throw std::runtime_error(error_msg.str());
    }
    // Sort actual and expected outputs
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    // Verify both are equal
    for (size_t i = 0; i < actual.size(); i++)
    {
        const auto& p = actual.at(i);
        const auto& q = expected.at(i);
        if (p.x() != q.x() || p.y() != q.y())
        {
            std::stringstream error_msg;
            error_msg << "Actual convex-hull is different than expected. "
                      << "Point at index " << i << " is " << p.toString()
                      << ", when " << q.toString() << ", was expected.\n";
            throw std::runtime_error(error_msg.str());
        }
    }
}

void testPointInsideConvexPolygon()
{
    using namespace gcransac::utils;
    // Construct a convex polygon
    std::vector<Point2D> convex_polygon;
    convex_polygon.emplace_back(0, 0);
    convex_polygon.emplace_back(3, 0);
    convex_polygon.emplace_back(3, 3);
    convex_polygon.emplace_back(0, 3);
    // Construct a point expected to be inside of convex polygon
    const Point2D point{1, 2};
    if (!pointInConvexPolygon(point, convex_polygon))
    {
        std::stringstream error_msg;
        error_msg << "2D-point (" << point.x() << ", " << point.y() << ") "
                  << "is not inside the convex polygon with vertices:";
        for (size_t i = 0; i < convex_polygon.size(); i++)
        {
            const auto& vert = convex_polygon.at(i);
            error_msg << " (" << vert.x() << ", " << vert.y() << ")";
        }
        error_msg << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

void testPointOutsideConvexPolygon()
{
    using namespace gcransac::utils;
    // Construct a convex polygon
    std::vector<Point2D> convex_polygon;
    convex_polygon.emplace_back(0, 0);
    convex_polygon.emplace_back(3, 0);
    convex_polygon.emplace_back(3, 3);
    convex_polygon.emplace_back(0, 3);
    // Construct a point expected to be inside of convex polygon
    const Point2D point{-1, 2};
    if (pointInConvexPolygon(point, convex_polygon))
    {
        std::stringstream error_msg;
        error_msg << "2D-point (" << point.x() << ", " << point.y() << ") "
                  << "is not outside the convex polygon with vertices:";
        for (size_t i = 0; i < convex_polygon.size(); i++)
        {
            const auto& vert = convex_polygon.at(i);
            error_msg << " (" << vert.x() << ", " << vert.y() << ")";
        }
        error_msg << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

void testPointOnEdgeOfConvexPolygon()
{
    using namespace gcransac::utils;
    // Construct a convex polygon
    std::vector<Point2D> convex_polygon;
    convex_polygon.emplace_back(0, 0);
    convex_polygon.emplace_back(3, 0);
    convex_polygon.emplace_back(3, 3);
    convex_polygon.emplace_back(0, 3);
    // Construct a point expected to be inside of convex polygon
    const Point2D point{1.5, 1.5};
    if (!pointInConvexPolygon(point, convex_polygon))
    {
        std::stringstream error_msg;
        error_msg << "2D-point (" << point.x() << ", " << point.y() << ") "
                  << "is not inside the convex polygon with vertices:";
        for (size_t i = 0; i < convex_polygon.size(); i++)
        {
            const auto& vert = convex_polygon.at(i);
            error_msg << " (" << vert.x() << ", " << vert.y() << ")";
        }
        error_msg << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

void testPointOnVertexOfConvexPolygon()
{
    using namespace gcransac::utils;
    // Construct a convex polygon
    std::vector<Point2D> convex_polygon;
    convex_polygon.emplace_back(0, 0);
    convex_polygon.emplace_back(3, 0);
    convex_polygon.emplace_back(3, 3);
    convex_polygon.emplace_back(0, 3);
    // Construct a point expected to be inside of convex polygon
    const Point2D point{3, 3};
    if (!pointInConvexPolygon(point, convex_polygon))
    {
        std::stringstream error_msg;
        error_msg << "2D-point (" << point.x() << ", " << point.y() << ") "
                  << "is not inside the convex polygon with vertices:";
        for (size_t i = 0; i < convex_polygon.size(); i++)
        {
            const auto& vert = convex_polygon.at(i);
            error_msg << " (" << vert.x() << ", " << vert.y() << ")";
        }
        error_msg << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

void testLineFromPointAndAngle()
{
    using namespace gcransac::utils;
    auto line = lineFromPointAndAngle(0.0, 0.0, 0.0);
    Eigen::Vector3d point{1.0, 0.0, 1.0}; 
    double dist = std::abs(line.dot(point));
    if (dist > 1e-9)
    {
        std::stringstream error_msg;
        error_msg << "Line intersecting (0, 0) and angle 0 did not intersect "
                     "(1, 0). The distance to it is " << dist << std::endl;
        throw std::runtime_error(error_msg.str());
    }
    point = {-1.0, 1.0, 1.0};
    dist = std::abs(line.dot(point));
    if (std::abs(dist - 1.0) > 1e-9)
    {
        std::stringstream error_msg;
        error_msg << "Line intersecting (0, 0) and angle 0 is not 1 unit of "
                     "distance distance away from (-1, 1). The distance is "
                  <<  dist << std::endl;
        throw std::runtime_error(error_msg.str());
    }
    auto other_line = lineFromPointAndAngle(1.0, 1.0, M_PI_2);
    auto intersection = line.cross(other_line);
    if (std::abs(intersection(2)) < 1e-9)
    {
        std::stringstream error_msg;
        error_msg << "Intersection of lines is a point at infinity.\n";
        throw std::runtime_error(error_msg.str());
    }
    intersection /= intersection(2);
    // subtract the expected intersection from the result
    Eigen::Vector3d expected{1.0, 0.0, 1.0};
    // check if result minus the expected intersection has zero length
    if ((intersection - expected).norm() > 1e-9)
    {
        std::stringstream error_msg;
        error_msg << "Intersection of lines is expected to be "
                     "(" << expected(0) << ", " << expected(1) << "). Received "
                     "(" << intersection(0) << ", " << intersection(1) << ").\n";
        throw std::runtime_error(error_msg.str());
    }
}

void testDegreesToRadians()
{
    using namespace gcransac::utils;
    constexpr double kEpsilon = 1e-9;
    constexpr double kDegs = 90.0;
    constexpr double kRads = M_PI_2;
    auto result = deg2rad(kDegs);
    if (std::abs(result - kRads) > kEpsilon)
    {
        std::stringstream error_msg;
        error_msg << "deg2rad(" << kDegs << ") returned " << result << ". "
                     "Expected " << kRads << ".\n";
        throw std::runtime_error(error_msg.str());
    }
    result = rad2deg(kRads);
    if (std::abs(result - kDegs) > kEpsilon)
    {
        std::stringstream error_msg;
        error_msg << "rad2deg(" << kRads << ") returned " << result << ". "
                     "Expected " << kDegs << ".\n";
        throw std::runtime_error(error_msg.str());
    }
    result = rad2deg(deg2rad(kDegs));
    if (std::abs(result - kDegs) > kEpsilon)
    {
        std::stringstream error_msg;
        error_msg << "rad2deg(deg2rad(" << kDegs << ")) returned " << result
                  << ". Expected " << kDegs << ".\n";
        throw std::runtime_error(error_msg.str());
    }
}

void testClipAngle()
{
    using namespace gcransac::utils;
    const std::array<double, 5> input = {-M_PI_2, 0.7861, 0.0, 2.7 * M_PI, -M_PI};
    const std::array<double, 5> expected = {1.5 * M_PI, 0.7861, 0.0, 0.7 * M_PI, M_PI};
    for (size_t i = 0; i < input.size(); i++)
    {
        auto result = clipAngle(input[i]);
        if (std::fabs(result - expected[i]) > 1e-9)
        {
            std::stringstream error_msg;
            error_msg << "clipAngle(" << input[i] << ") expected to return "
                      << expected[i] << ", but returned " << result
                      << " instead.\n";
            throw std::runtime_error(error_msg.str());
        }
    }
}

void testMinAngleDiff()
{
    using namespace gcransac::utils;
    std::vector<std::pair<double, double>> input;
    std::vector<double> expected;
    
    input.emplace_back(0.0, 0.0);
    expected.push_back(0.0);

    input.emplace_back(-M_PI_2, 0.0);
    expected.push_back(M_PI_2);

    input.emplace_back(-3.48 * M_PI, 7.41 * M_PI);
    expected.push_back(0.89 * M_PI);

    input.emplace_back(0.25 * M_PI, 1.25 * M_PI);
    expected.push_back(M_PI);

    for (size_t i = 0; i < input.size(); i++)
    {
        double angle1 = input.at(i).first;
        double angle2 = input.at(i).second;
        double result = minAngleDiff(angle1, angle2);
        if (std::fabs(result - expected.at(i)) > 1e-9)
        {
            std::stringstream error_msg;
            error_msg << "minAngleDiff(" << angle1 << ", " << angle2 << ") "
                      << "expected to return " << expected.at(i) << ", but "
                      << "returned " << result << " instead.\n";
            throw std::runtime_error(error_msg.str());
        }
    }
}

void testLinesAnglesDiff()
{
    using namespace gcransac::utils;
    std::vector<std::pair<double, double>> input;
    std::vector<double> expected;

    input.emplace_back(0.0, 0.0);
    expected.push_back(0.0);

    input.emplace_back(-M_PI_2, 0.0);
    expected.push_back(M_PI_2);

    input.emplace_back(0.52 * M_PI, 1.49 * M_PI);
    expected.push_back(0.03 * M_PI);

    input.emplace_back(1.49 * M_PI, 0.52 * M_PI);
    expected.push_back(0.03 * M_PI);

    input.emplace_back(-3.48 * M_PI, 5.60 * M_PI);
    expected.push_back(0.08 * M_PI);

    input.emplace_back(0.25 * M_PI, 1.25 * M_PI);
    expected.push_back(0.0);

    for (size_t i = 0; i < input.size(); i++)
    {
        double angle1 = input.at(i).first;
        double angle2 = input.at(i).second;
        double result = linesAnglesDiff(angle1, angle2);
        if (std::fabs(result - expected.at(i)) > 1e-9)
        {
            std::stringstream error_msg;
            error_msg << "linesAnglesDiff2(" << angle1 << ", " << angle2 << ") "
                      << "expected to return " << expected.at(i) << ", but "
                      << "returned " << result << " instead.\n";
            throw std::runtime_error(error_msg.str());
        }
    }
}

void testNChoose2()
{
    using namespace gcransac::utils;
    size_t expected = 15;
    auto result = nChoose2(6);
    if (result != expected)
    {
        std::stringstream error_msg;
        error_msg << "nChoose2(6)) returned " << result << ". Expected "
                  << expected << ".\n";
        throw std::runtime_error(error_msg.str());
    }
    expected = 0;
    result = nChoose2(1);
    if (result != expected)
    {
        std::stringstream error_msg;
        error_msg << "nChoose2(1)) returned " << result << ". Expected "
                  << expected << ".\n";
        throw std::runtime_error(error_msg.str());
    }
    expected = 0;
    result = nChoose2(0);
    if (result != expected)
    {
        std::stringstream error_msg;
        error_msg << "nChoose2(0)) returned " << result << ". Expected "
                  << expected << ".\n";
        throw std::runtime_error(error_msg.str());
    }
}

void runTest(const std::string& desc, void (*test)())
{
    static size_t index{0};
    try
    {
        test();
        std::cout << "\033[32m" << "SUCCESS" << "\033[0m"
                  << " Test #" << ++index << ": " << desc << "\n";
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "\033[31m"<< "FAILURE" << "\033[0m"
                  << " Test #" << ++index << ": " << desc << "\n"  
                  << "\t\t" << e.what() << "\n";
    }
}

int main()
{
    runTest("collinear positive case", testCollinearPointsPositiveCase);
    runTest("collinear negative case" ,testCollinearPointsNegativeCase);
    runTest("convex-hull", testConvexHullAlgo);
    runTest("degenerate 1D convex-hull", testDegenerateConvexHull1D);
    runTest("degenerate 2D convex-hull", testDegenerateConvexHull2D);
    runTest("point inside convex-polygon", testPointInsideConvexPolygon);
    runTest("point outside convex-polygon", testPointOutsideConvexPolygon);
    runTest("point on edge of convex-polygon", testPointOnEdgeOfConvexPolygon);
    runTest("point on vertex of convex-polygon", testPointOnVertexOfConvexPolygon);
    runTest("Line from point and angle", testLineFromPointAndAngle);
    runTest("degrees to radians", testDegreesToRadians);
    runTest("clip angle", testClipAngle);
    runTest("minimal angle difference", testMinAngleDiff);
    runTest("lines angles difference", testLinesAnglesDiff);
    runTest("n choose 2", testNChoose2);
}
