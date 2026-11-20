# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(quacs_embed VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The Python that QUACS is installed in. Its interpreter library is linked into
# quacs_embed, and its path is compiled in so that the embedded interpreter finds QUACS.
find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Embed)

# An installed pybind11 is used if there is one. Otherwise it is downloaded.
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.4
  FIND_PACKAGE_ARGS 3.0 CONFIG
)
FetchContent_MakeAvailable(pybind11)

add_library(quacs_embed SHARED quacs_embed.cpp)
target_include_directories(quacs_embed PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(quacs_embed PRIVATE pybind11::embed)
target_compile_definitions(quacs_embed PRIVATE
  QUACS_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
)
set_target_properties(quacs_embed PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER quacs_embed.h
  # Lets the installed library find the Python library it was built with.
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS quacs_embed
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
configure_file(quacs_embed.pc.in quacs_embed.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/quacs_embed.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

include(CTest)
if(BUILD_TESTING)
  # A host application written in C that runs the QUACS conformance checks.
  add_executable(test_host test_host.c)
  target_link_libraries(test_host PRIVATE quacs_embed)
  add_test(NAME conformance
    COMMAND test_host ${CMAKE_CURRENT_SOURCE_DIR}/test_config.py
  )
endif()