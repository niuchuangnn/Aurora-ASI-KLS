cmake_minimum_required(VERSION 3.3)

project(selective_search)

execute_process(COMMAND python2.7-config --cflags  OUTPUT_VARIABLE PYTHON_CFLAGS  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python2.7-config --ldflags OUTPUT_VARIABLE PYTHON_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
set(BOOST_LIBS "-lboost_system -lboost_python -lboost_numpy")

set(SUPPRESS_WARNING "-Wall -Wno-unused-function -Wno-unused-variable -Wno-unused-local-typedefs")

include_directories("segment")
set(CMAKE_CXX_FLAGS "-fPIC -O3 -std=c++11 ${PYTHON_CFLAGS} ${SUPPRESS_WARNING}")
set(CMAKE_SHARED_LINKER_FLAGS "${PYTHON_LDFLAGS} -shared -fPIC")

add_library(segment SHARED segment_py.cpp)
target_link_libraries(segment ${BOOST_LIBS})
set_target_properties(segment PROPERTIES PREFIX "")
