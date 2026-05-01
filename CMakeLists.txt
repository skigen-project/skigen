cmake_minimum_required(VERSION 3.20)
project(skigen
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Energy-efficient header-only ML library built on Eigen"
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Eigen dependency
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    # Fall back to source-tree Eigen (e.g. sibling directory)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../eigen/Eigen/Core")
        add_library(Eigen3_source INTERFACE)
        target_include_directories(Eigen3_source INTERFACE
            "${CMAKE_CURRENT_SOURCE_DIR}/../eigen"
        )
        add_library(Eigen3::Eigen ALIAS Eigen3_source)
        message(STATUS "Using Eigen from source tree: ${CMAKE_CURRENT_SOURCE_DIR}/../eigen")
    else()
        message(FATAL_ERROR
            "Eigen3 not found. Install Eigen3 or place the Eigen source tree "
            "as a sibling directory at ../eigen")
    endif()
endif()

# ---------------------------------------------------------------------------
# Header-only interface library
# ---------------------------------------------------------------------------
add_library(Skigen INTERFACE)
add_library(Skigen::Skigen ALIAS Skigen)

target_include_directories(Skigen INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:.>
)

target_link_libraries(Skigen INTERFACE Eigen3::Eigen)
target_compile_features(Skigen INTERFACE cxx_std_23)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SKIGEN_BUILD_TESTS      "Build unit tests"      ON)
option(SKIGEN_BUILD_BENCHMARKS "Build benchmarks"       OFF)
option(SKIGEN_BUILD_EXAMPLES   "Build example programs" OFF)

# ---------------------------------------------------------------------------
# Strict warnings for development targets
# ---------------------------------------------------------------------------
set(SKIGEN_WARNING_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(SKIGEN_WARNING_FLAGS -Wall -Wextra -Wpedantic)
elseif(MSVC)
    set(SKIGEN_WARNING_FLAGS /W4)
endif()

# ---------------------------------------------------------------------------
# Sub-directories
# ---------------------------------------------------------------------------
if(SKIGEN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SKIGEN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(SKIGEN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS Skigen EXPORT SkigenTargets)

install(DIRECTORY Skigen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(Eigen3_FOUND)
    install(EXPORT SkigenTargets
        FILE SkigenTargets.cmake
        NAMESPACE Skigen::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Skigen
    )
endif()
