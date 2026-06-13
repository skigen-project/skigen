cmake_minimum_required(VERSION 3.20)
project(skigen
    VERSION 1.1.0
    LANGUAGES CXX
    DESCRIPTION "Energy-efficient header-only ML library built on Eigen"
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Eigen dependency
# ---------------------------------------------------------------------------
# If a parent project already provides Eigen3::Eigen, skip discovery
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        # Fall back to source-tree Eigen (in-tree or sibling directory)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/eigen/Eigen/Core")
            set(_eigen_dir "${CMAKE_CURRENT_SOURCE_DIR}/eigen")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../eigen/Eigen/Core")
            set(_eigen_dir "${CMAKE_CURRENT_SOURCE_DIR}/../eigen")
        endif()
        if(_eigen_dir)
            add_library(Eigen3_source INTERFACE)
            target_include_directories(Eigen3_source INTERFACE "${_eigen_dir}")
            add_library(Eigen3::Eigen ALIAS Eigen3_source)
            message(STATUS "Using Eigen from source tree: ${_eigen_dir}")
        else()
            message(FATAL_ERROR
                "Eigen3 not found. Install Eigen3 or place the Eigen source tree "
                "at ./eigen or ../eigen")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Header-only interface library
# ---------------------------------------------------------------------------
add_library(Skigen INTERFACE)
add_library(Skigen::Skigen ALIAS Skigen)

target_link_libraries(Skigen INTERFACE Eigen3::Eigen)
target_compile_features(Skigen INTERFACE cxx_std_23)

# The SKIGEN_PARAMS reflection macro feeds parenthesised tuples to a
# variadic FOR_EACH expansion. MSVC's traditional preprocessor splits
# the inner commas as if they were macro arg separators, breaking the
# expansion; the conformant preprocessor (enabled by /Zc:preprocessor,
# available since VS 2017 16.5) handles it correctly. The flag also
# brings MSVC into line with __VA_OPT__ / __VA_ARGS__ behaviour the
# rest of the toolchain assumes.
if(MSVC)
    target_compile_options(Skigen INTERFACE /Zc:preprocessor)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SKIGEN_BUILD_TESTS      "Build unit tests"      ON)
option(SKIGEN_BUILD_BENCHMARKS "Build benchmarks"       OFF)
option(SKIGEN_BUILD_EXAMPLES   "Build example programs" OFF)
option(SKIGEN_WITH_PLOT        "Enable optional SkigenPlot integration for examples" OFF)

# ---------------------------------------------------------------------------
# Optional SkigenPlot dependency (used only by examples that opt in)
# ---------------------------------------------------------------------------
# Skigen itself stays strictly header-only. SkigenPlot is a compiled Qt/RHI
# library and remains an opt-in dependency that only a handful of example
# targets link against when SKIGEN_WITH_PLOT=ON.
if(SKIGEN_WITH_PLOT AND NOT TARGET SkigenPlot::SkigenPlot)
    find_package(SkigenPlot QUIET)
    if(NOT SkigenPlot_FOUND)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/skigen-plot/CMakeLists.txt")
            set(_skigen_plot_dir "${CMAKE_CURRENT_SOURCE_DIR}/skigen-plot")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../skigen-plot/CMakeLists.txt")
            set(_skigen_plot_dir "${CMAKE_CURRENT_SOURCE_DIR}/../skigen-plot")
        endif()
        if(_skigen_plot_dir)
            message(STATUS "Using SkigenPlot from source tree: ${_skigen_plot_dir}")
            add_subdirectory("${_skigen_plot_dir}" skigen-plot-build EXCLUDE_FROM_ALL)
        else()
            message(FATAL_ERROR
                "SKIGEN_WITH_PLOT=ON but SkigenPlot was not found. "
                "Install SkigenPlot or place its source tree at "
                "./skigen-plot or ../skigen-plot.")
        endif()
    endif()
endif()

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
# Documentation (Doxygen API docs)
# ---------------------------------------------------------------------------
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs_api
        COMMAND doxygen ${CMAKE_SOURCE_DIR}/doc/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc
        COMMENT "Generating Doxygen XML for API reference"
    )
else()
    message(STATUS "Doxygen not found — docs_api target disabled")
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_INCLUDEDIR}/skigen"
    CACHE PATH "Header install directory (relative to CMAKE_INSTALL_PREFIX)")

set(CMAKEPACKAGE_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/skigen/cmake"
    CACHE PATH "CMake package config install directory")

# Make paths relative to CMAKE_INSTALL_PREFIX if absolute
foreach(var INCLUDE_INSTALL_DIR CMAKEPACKAGE_INSTALL_DIR)
    if(IS_ABSOLUTE "${${var}}")
        file(RELATIVE_PATH "${var}" "${CMAKE_INSTALL_PREFIX}" "${${var}}")
    endif()
endforeach()

# Update the interface target to use the install path
target_include_directories(Skigen INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

# Install headers
install(DIRECTORY Skigen
    DESTINATION ${INCLUDE_INSTALL_DIR}
    COMPONENT Devel
)

# CMake package export — only when Eigen was found via find_package
# (source-tree Eigen3_source target is not exportable)
if(Eigen3_FOUND)
    install(TARGETS Skigen EXPORT SkigenTargets)

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SkigenConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenConfig.cmake"
        INSTALL_DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(EXPORT SkigenTargets
        FILE SkigenTargets.cmake
        NAMESPACE Skigen::
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenConfigVersion.cmake"
        DESTINATION ${CMAKEPACKAGE_INSTALL_DIR}
    )

    export(EXPORT SkigenTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/SkigenTargets.cmake"
        NAMESPACE Skigen::
    )
else()
    message(STATUS "Eigen from source tree — CMake package export disabled. "
                   "Install Eigen3 system-wide for find_package(Skigen) support.")
endif()

# Uninstall target
if(NOT TARGET uninstall AND PROJECT_IS_TOP_LEVEL)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SkigenUninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/SkigenUninstall.cmake"
        @ONLY
    )
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/SkigenUninstall.cmake"
    )
endif()

# Print install summary
message(STATUS "")
message(STATUS "Skigen v${PROJECT_VERSION} — install layout:")
message(STATUS "  Headers:    \${CMAKE_INSTALL_PREFIX}/${INCLUDE_INSTALL_DIR}/Skigen/")
message(STATUS "  CMake pkg:  \${CMAKE_INSTALL_PREFIX}/${CMAKEPACKAGE_INSTALL_DIR}/")
message(STATUS "")
