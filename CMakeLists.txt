cmake_minimum_required(VERSION 2.8.7)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AppendCompilerFlags)
set(CMAKE_OSX_ARCHITECTURES "x86_64")

project(Ring)
set(CMAKE_PROJECT_NAME "Ring")

#Change to "Debug" when you need it
set(CMAKE_BUILD_TYPE "Debug")

# C++11 compiler Check
if(NOT CMAKE_CXX_COMPILER_VERSION) # work around for cmake versions smaller than 2.8.10
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
endif()
if(CMAKE_CXX_COMPILER MATCHES ".*clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()
if( (CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 4.7) OR
(CMAKE_COMPILER_IS_CLANGXX AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.2))
    message(FATAL_ERROR "Your C++ compiler does not support C++11. Please install g++ 4.7 (or greater) or clang 3.2 (or greater)")
else()
    message(STATUS "Compiler is recent enough to support C++11.")
endif()
if( CMAKE_COMPILER_IS_GNUCXX )
    append_cxx_compiler_flags("-fpermissive -std=c++11 -Wall -Wextra " "GCC" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-msse4.2 -O3 -ffast-math -funroll-loops -fno-omit-frame-pointer -g" "GCC" CMAKE_CXX_FLAGS_RELEASE)
else()
    append_cxx_compiler_flags("-fpermissive -std=c++11" "CLANG" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-stdlib=libc++" "CLANG" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-msse4.2 -O3  -ffast-math -funroll-loops -DNDEBUG" "CLANG" CMAKE_CXX_FLAGS_RELEASE)
endif()
include(CheckSSE)
FindSSE ()
if( SSE4_2_FOUND )
    if( CMAKE_COMPILER_IS_GNUCXX )
        append_cxx_compiler_flags("-msse4.2" "GCC" CMAKE_CXX_FLAGS)
    else()
        append_cxx_compiler_flags("-msse4.2" "CLANG" CMAKE_CXX_FLAGS)
    endif()
    message(STATUS "CPU does support SSE4.2.")
else()
    message(STATUS "CPU does NOT support SSE4.2")
endif()

include_directories(~/include
                    ${CMAKE_HOME_DIRECTORY}/include)

link_directories(~/lib)


add_executable(build-index src/build-index.cpp)
target_link_libraries(build-index sdsl divsufsort divsufsort64)


add_executable(query-index src/query-index.cpp)
target_compile_definitions(query-index PUBLIC ADAPTIVE=1)
target_link_libraries(query-index sdsl divsufsort divsufsort64)

add_executable(query-index-fixed src/query-index.cpp)
target_compile_definitions(query-index-fixed PUBLIC ADAPTIVE=0)
target_link_libraries(query-index-fixed sdsl divsufsort divsufsort64)

add_executable(query-index-random src/query-index.cpp)
target_compile_definitions(query-index-random PUBLIC ADAPTIVE=0 VEO_RANDOM=1)
target_link_libraries(query-index-random sdsl divsufsort divsufsort64)

add_executable(build-index-uring src/build-index-uring.cpp)
target_link_libraries(build-index-uring sdsl divsufsort divsufsort64)

add_executable(query-index-uring src/query-index-uring.cpp)
target_compile_definitions(query-index-uring PUBLIC ADAPTIVE=1)
target_link_libraries(query-index-uring sdsl divsufsort divsufsort64)

add_executable(query-index-uring-fixed src/query-index-uring.cpp)
target_compile_definitions(query-index-uring-fixed PUBLIC ADAPTIVE=0)
target_link_libraries(query-index-uring-fixed sdsl divsufsort divsufsort64)

add_executable(query-index-veo-intersection src/query-index-veo-intersection.cpp)
target_compile_definitions(query-index-veo-intersection PUBLIC ADAPTIVE=1)
target_link_libraries(query-index-veo-intersection sdsl divsufsort divsufsort64)

add_executable(query-index-veo-intersection-fixed src/query-index-veo-intersection.cpp)
target_compile_definitions(query-index-veo-intersection-fixed PUBLIC ADAPTIVE=0)
target_link_libraries(query-index-veo-intersection-fixed sdsl divsufsort divsufsort64)

add_executable(query-index-uring-veo-intersection src/query-index-uring-veo-intersection.cpp)
target_compile_definitions(query-index-uring-veo-intersection PUBLIC ADAPTIVE=1)
target_link_libraries(query-index-uring-veo-intersection sdsl divsufsort divsufsort64)

add_executable(query-index-uring-veo-intersection-fixed src/query-index-uring-veo-intersection.cpp)
target_compile_definitions(query-index-uring-veo-intersection-fixed PUBLIC ADAPTIVE=0)
target_link_libraries(query-index-uring-veo-intersection-fixed sdsl divsufsort divsufsort64)

add_executable(query-index-veo-prob-depth src/query-index-veo-prob-depth.cpp)
target_link_libraries(query-index-veo-prob-depth sdsl divsufsort divsufsort64)

add_executable(query-index-veo-probability src/query-index-veo-probability.cpp)
target_compile_definitions(query-index-veo-probability PUBLIC ADAPTIVE=1)
target_link_libraries(query-index-veo-probability sdsl divsufsort divsufsort64)

add_executable(query-index-veo-probability-fixed src/query-index-veo-probability.cpp)
target_compile_definitions(query-index-veo-probability-fixed PUBLIC ADAPTIVE=0)
target_link_libraries(query-index-veo-probability-fixed sdsl divsufsort divsufsort64)

add_executable(check src/check.cpp)
target_link_libraries(check sdsl divsufsort divsufsort64)

add_executable(check_veo src/check_veo.cpp)
target_link_libraries(check_veo sdsl divsufsort divsufsort64)

add_executable(proba src/proba.cpp)
target_link_libraries(proba sdsl divsufsort divsufsort64)
