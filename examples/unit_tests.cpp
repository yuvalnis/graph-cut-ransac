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
}
