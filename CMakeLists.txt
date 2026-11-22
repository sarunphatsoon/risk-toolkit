cmake_minimum_required(VERSION 3.18)
project(credit_mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(creditmc STATIC src/engine.cpp src/market_risk.cpp src/backtest.cpp)
target_include_directories(creditmc PUBLIC include)
target_link_libraries(creditmc PUBLIC Threads::Threads)
set_target_properties(creditmc PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(BUILD_PYTHON "Build the Python extension module" ON)
option(BUILD_TESTS "Build the unit tests" ON)

if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(creditmc_core src/bindings.cpp)
  target_link_libraries(creditmc_core PRIVATE creditmc)
  set_target_properties(creditmc_core PROPERTIES
   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
endif()

if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1)
  FetchContent_MakeAvailable(Catch2)

  enable_testing()
  add_executable(tests tests/test_engine.cpp tests/test_market_risk.cpp tests/test_backtest.cpp tests/test_batch.cpp tests/test_sliding_window.cpp)
  target_link_libraries(tests PRIVATE creditmc Catch2::Catch2WithMain)

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(tests)
endif()