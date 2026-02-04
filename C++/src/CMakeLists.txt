cmake_minimum_required(VERSION 3.15)
project(run_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(run_python test.cpp)

add_custom_command(TARGET run_python PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../../python/test.py
        $<TARGET_FILE_DIR:run_python>
)

add_custom_command(TARGET run_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/../../models/yolo11n.pt
    $<TARGET_FILE_DIR:run_python>)

add_custom_command(TARGET run_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/../../python/siyi_a8_calib.npz
    $<TARGET_FILE_DIR:run_python>)