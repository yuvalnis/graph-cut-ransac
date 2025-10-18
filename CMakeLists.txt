cmake_minimum_required(VERSION 3.8)
project(GraphCutRANSAC LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" OFF)
option(UNIT_TESTS "Build Unit Tests" ON)
option(TEST_THREE_SIFT_SCALE_ONLY "Test Three SIFT Scale Only" OFF)
option(TEST_TWO_SIFT "Test Two SIFT" OFF)
option(TEST_NON_ID_TWO_SIFT "Test Non Identity TWO SIFT" ON)
option(TEST_NON_ID_THREE_SIFT "Test Non Identity THREE SIFT" ON)
option(TEST_THREE_SIFT_CUSTOM_DATA "Test THREE SIFT Custom Data" ON)
option(TEST_TWO_SIFT_CUSTOM_DATA "Test TWO SIFT Custom Data" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(OpenCV)
if (OPENCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
  	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (OPENCV_FOUND)

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (OPENMP_FOUND)

# SET(CMAKE_BUILD_TYPE "DEBUG")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})


SET(SOURCE_DIR "src/pygcransac")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR}/include)


# Set header files for the library
file(GLOB_RECURSE HDRS_GCRANSAC
	"${SOURCE_DIR}/include/*.h"
	"${SOURCE_DIR}/include/*.hpp"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GCRANSAC 
	"${SOURCE_DIR}/include/*.cpp"
)


# Generate python module
add_subdirectory(lib/pybind11)

pybind11_add_module(pygcransac ${SOURCE_DIR}/src/bindings.cpp ${SOURCE_DIR}/src/gcransac_python.cpp  ${HDRS_GCRANSAC} ${SRCS_GCRANSAC} )
target_link_libraries(pygcransac PRIVATE  ${OpenCV_LIBS} Eigen3::Eigen)


add_library(${PROJECT_NAME} STATIC ${HDRS_GCRANSAC} ${SRCS_GCRANSAC})

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
)

target_link_libraries (${PROJECT_NAME}
	Eigen3::Eigen)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(SampleProject
		examples/cpp_example.cpp)
		
	target_link_libraries (SampleProject 
		Eigen3::Eigen)
		
	target_link_libraries(SampleProject 
		${OpenCV_LIBS}
		${PROJECT_NAME}
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)

# ==============================================================================
# Unit Tests with Google Test
# ==============================================================================
if (UNIT_TESTS)
	# Setup Google Test
	include(FetchContent)
	FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG v1.14.0
	)
	# For Windows: Prevent overriding the parent project's compiler/linker settings
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)
	enable_testing()
	include(GoogleTest)

	# Build unit tests executable
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(unit_tests tests/unit_tests.cpp)
	target_sources(unit_tests PRIVATE ${GCRANSAC_PYTHON_SRC})
	
	target_link_libraries(unit_tests 
		PRIVATE
		Eigen3::Eigen
		${OpenCV_LIBS}
		${PROJECT_NAME}
		GTest::gtest_main
	)

	add_dependencies(unit_tests ${PROJECT_NAME})

	# Enable test discovery for CTest integration
	gtest_discover_tests(unit_tests)
endif()

if (TEST_THREE_SIFT_SCALE_ONLY)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(
		test_three_sift_scale_only
		examples/test_three_sift_scale_only.cpp
	)

	target_sources(
		test_three_sift_scale_only
		PRIVATE ${GCRANSAC_PYTHON_SRC}
	)
		
	target_link_libraries (test_three_sift_scale_only 
		Eigen3::Eigen)
		
	target_link_libraries(test_three_sift_scale_only 
		${OpenCV_LIBS}
	)

	target_link_libraries(test_three_sift_scale_only 
		debug 
		${PROJECT_NAME}
	)

	add_dependencies(test_three_sift_scale_only ${PROJECT_NAME})
endif()

if (TEST_TWO_SIFT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(
		test_two_sift
		examples/test_two_sift.cpp
		${SOURCE_DIR}/src/gcransac_python.cpp
	)

	target_link_libraries (test_two_sift 
		Eigen3::Eigen)
		
	target_link_libraries(test_two_sift 
		${OpenCV_LIBS}
	)

	target_link_libraries(test_two_sift 
		debug 
		${PROJECT_NAME}
	)

	add_dependencies(test_two_sift ${PROJECT_NAME})
endif()

if (TEST_NON_ID_TWO_SIFT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(
		test_non_identity_homography_two_sift
		examples/test_non_identity_homography_two_sift.cpp
		${SOURCE_DIR}/src/gcransac_python.cpp
	)

	target_link_libraries (test_non_identity_homography_two_sift 
		Eigen3::Eigen)
		
	target_link_libraries(test_non_identity_homography_two_sift 
		${OpenCV_LIBS}
	)

	target_link_libraries(test_non_identity_homography_two_sift 
		debug 
		${PROJECT_NAME}
	)

	add_dependencies(test_non_identity_homography_two_sift ${PROJECT_NAME})
endif()

if (TEST_NON_ID_THREE_SIFT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(
		test_non_identity_homography_three_sift_scale_only
		examples/test_non_identity_homography_three_sift_scale_only.cpp
		${SOURCE_DIR}/src/gcransac_python.cpp
	)

	target_link_libraries (test_non_identity_homography_three_sift_scale_only 
		Eigen3::Eigen)
		
	target_link_libraries(test_non_identity_homography_three_sift_scale_only 
		${OpenCV_LIBS}
	)

	target_link_libraries(test_non_identity_homography_three_sift_scale_only 
		debug 
		${PROJECT_NAME}
	)

	add_dependencies(test_non_identity_homography_three_sift_scale_only ${PROJECT_NAME})
endif()

if (TEST_THREE_SIFT_CUSTOM_DATA)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(
		test_three_sift_custom_data
		examples/test_three_sift_scale_only_custom_data.cpp
		${SOURCE_DIR}/src/gcransac_python.cpp
	)

	target_link_libraries (test_three_sift_custom_data 
		Eigen3::Eigen)
		
	target_link_libraries(test_three_sift_custom_data 
		${OpenCV_LIBS}
	)

	target_link_libraries(test_three_sift_custom_data 
		debug 
		${PROJECT_NAME}
	)

	add_dependencies(test_three_sift_custom_data ${PROJECT_NAME})
endif()

if (TEST_TWO_SIFT_CUSTOM_DATA)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

	add_executable(
		test_two_sift_custom_data
		examples/test_two_sift_custom_data.cpp
		${SOURCE_DIR}/src/gcransac_python.cpp
	)

	target_link_libraries (test_two_sift_custom_data 
		Eigen3::Eigen)
		
	target_link_libraries(test_two_sift_custom_data 
		${OpenCV_LIBS}
	)

	target_link_libraries(test_two_sift_custom_data 
		debug 
		${PROJECT_NAME}
	)

	add_dependencies(test_two_sift_custom_data ${PROJECT_NAME})
endif()
