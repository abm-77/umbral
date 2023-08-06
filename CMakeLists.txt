cmake_minimum_required(VERSION 3.24)
project(umbral)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_FLAGS "-g")
set(CMAKE_CXX_FLAGS "-g")

include("${CMAKE_CURRENT_LIST_DIR}/build-utils.cmake")


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bld)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(GLSL_VALIDATOR "glslc")

add_compile_definitions(ENABLE_ASSERT=1)
add_compile_definitions(DEBUG=1)

# INTERNAL DEPENDENCIES
include_directories(${PROJECT_SOURCE_DIR})
set(UMBRAL_COMMON_DEPS umbral-internal)
umk_static_library(NAME umbral-internal
                  SRCS  ${CMAKE_CURRENT_LIST_DIR}/src/core/umb_mem.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/src/core/internal.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/src/sys/umb_app.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/src/sys/umb_file.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/src/sys/umb_window.cpp
                        ${CMAKE_CURRENT_LIST_DIR}/src/gfx/umb_vk.cpp
)

# EXTERNAL DEPENDENCIES 
find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${VulkanMemoryAllocator_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

umk_binary(NAME ${PROJECT_NAME}  
           SRCS ${CMAKE_SOURCE_DIR}/src/main.cpp
           DEPS umbral-internal ${SDL2_LIBRARIES} ${Vulkan_LIBRARIES} glm::glm)


 file(GLOB_RECURSE shader_src "${PROJECT_SOURCE_DIR}/gfx/shaders/*.vert" "${PROJECT_SOURCE_DIR}/gfx/shaders/*.frag")
 foreach(GLSL ${shader_src})
     get_filename_component(FILE_NAME ${GLSL} NAME)
     set(SPIRV "${CMAKE_CURRENT_LIST_DIR}/res/shaders/${FILE_NAME}.spv")
     add_custom_command(
         OUTPUT ${SPIRV}
         COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_LIST_DIR}/res/shaders/"
         COMMAND ${GLSL_VALIDATOR} ${GLSL} -o ${SPIRV}
         DEPENDS ${GLSL}
     )
     list(APPEND SPIRV_BINARY_FILES ${SPIRV})
 endforeach(GLSL)

add_custom_target(shaders DEPENDS ${SPIRV_BINARY_FILES})
add_dependencies(${PROJECT_NAME} shaders)
