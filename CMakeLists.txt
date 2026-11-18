cmake_minimum_required(VERSION 3.16)
project(MiniTensor LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)      # Release = -O3 -DNDEBUG, as required by the common test specification
endif()
option(MT_NATIVE "Use -march=native (fastest, but tied to this CPU)" ON)
option(MT_SANITIZE "Build test drivers with AddressSanitizer + UBSan (for TC46-TC50)" OFF)

find_package(OpenMP)
add_library(minitensor INTERFACE)
target_include_directories(minitensor INTERFACE include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(minitensor INTERFACE OpenMP::OpenMP_CXX)
endif()
target_compile_options(minitensor INTERFACE -O3)
if(MT_NATIVE)
  target_compile_options(minitensor INTERFACE -march=native)
endif()
if(MT_SANITIZE)
  target_compile_options(minitensor INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(minitensor INTERFACE -fsanitize=address,undefined)
endif()

add_executable(tests tests/test_tensor.cpp)           # quick unit tests (keeps assert() active even in Release)
target_link_libraries(tests minitensor)
add_executable(common_test tests/common_test.cpp)     # TC01-TC50 from the Common Test Specification
target_link_libraries(common_test minitensor)
add_executable(demo examples/demo.cpp)
target_link_libraries(demo minitensor)
add_executable(bench benchmarks/bench.cpp)            # simple naive-vs-fast matmul demo
target_link_libraries(bench minitensor)
add_executable(spec_bench benchmarks/spec_bench.cpp)  # PB01-PB12 + scaling runs from the specification
target_link_libraries(spec_bench minitensor)

enable_testing()
add_test(NAME unit_tests COMMAND tests)
add_test(NAME common_spec_tests COMMAND common_test TEAM ${CMAKE_BINARY_DIR}/functional_results.csv)
