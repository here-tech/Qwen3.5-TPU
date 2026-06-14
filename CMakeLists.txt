cmake_minimum_required(VERSION 3.10)
project(Qwen3VL)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/data/qwen/LLM-TPU/support/include)
link_directories(/data/qwen/LLM-TPU/support/lib_soc)

add_definitions(--std=c++17 -fPIC -Wall -Werror)
set(Python3_ROOT_DIR "/data/qwen3.5_env")
set(PYTHON_EXECUTABLE "/data/qwen3.5_env/bin/python3.10")
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
set(pybind11_DIR "/data/qwen3.5_env/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(chat chat.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)

