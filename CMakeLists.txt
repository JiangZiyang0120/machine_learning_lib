cmake_minimum_required(VERSION 3.20)
project(machine_learning)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)

#include_directories("${PROJECT_SOURCE_DIR}"/include/util)
# 把 util 子目录编译成 library，最终会生成 libutil.a 文件
#aux_source_directory(./util UTIL_SRC)
#add_library(util ${UTIL_SRC})

include_directories("${PROJECT_SOURCE_DIR}/include/util")    # includes ./include/util/*.h

# 生成可执行文件 test
add_executable(test test/test.cpp)
# 添加链接库 util 等
target_link_libraries(test util)

add_executable(machine_learning main.cpp include/util/ml_base_model.h)
