cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ========================================
# OptiX Configuration
# ========================================

# IMPORTANT: First, we need to set the OptiX location before including FindOptiX
# The OptiX SDK you downloaded should have an 'include' folder at its root
# If the structure is different, you may need to adjust this path
set(OptiX_ROOT_DIR "${CMAKE_SOURCE_DIR}/external/OptiX SDK 9.0.0")

# Check if the include directory exists in the expected location
if(EXISTS "${OptiX_ROOT_DIR}/include")
    set(OptiX_INCLUDE_DIR "${OptiX_ROOT_DIR}/include")
elseif(EXISTS "${OptiX_ROOT_DIR}/SDK/include")
    set(OptiX_INCLUDE_DIR "${OptiX_ROOT_DIR}/SDK/include")
elseif(EXISTS "${OptiX_ROOT_DIR}/optix/include")
    set(OptiX_INCLUDE_DIR "${OptiX_ROOT_DIR}/optix/include")
else()
    # If not found, let's look for optix.h recursively
    file(GLOB_RECURSE OPTIX_H_PATH "${OptiX_ROOT_DIR}/**/optix.h")
    if(OPTIX_H_PATH)
        get_filename_component(OptiX_INCLUDE_DIR "${OPTIX_H_PATH}" DIRECTORY)
        message(STATUS "Found optix.h at: ${OPTIX_H_PATH}")
    else()
        message(FATAL_ERROR "Could not find optix.h in ${OptiX_ROOT_DIR}. Please check your OptiX SDK installation.")
    endif()
endif()

# Set the variables that FindOptiX.cmake expects
set(OptiX_INSTALL_DIR "${OptiX_ROOT_DIR}" CACHE PATH "Path to OptiX installed location." FORCE)
set(OPTIX_ROOT_DIR "${OptiX_ROOT_DIR}" CACHE PATH "Path to OptiX installed location." FORCE)

# Add OptiX SDK CMake modules to the module path
if(EXISTS "${OptiX_ROOT_DIR}/SDK/CMake")
    set(CMAKE_MODULE_PATH 
        "${OptiX_ROOT_DIR}/SDK/CMake"
        ${CMAKE_MODULE_PATH}
    )
endif()

# Now try to find OptiX
# If FindOptiX.cmake exists in the SDK, use it; otherwise, we'll set the variables manually
if(EXISTS "${OptiX_ROOT_DIR}/SDK/CMake/FindOptiX.cmake")
    # Set hints for FindOptiX
    set(OptiX_LOCATION "${OptiX_ROOT_DIR}")
    set(OPTIX_LOCATION "${OptiX_ROOT_DIR}")
    
    find_package(OptiX REQUIRED)
else()
    # Manually set OptiX variables if FindOptiX.cmake doesn't exist or doesn't work
    message(STATUS "Manually configuring OptiX paths")
    set(OptiX_FOUND TRUE)
    set(OptiX_INCLUDE "${OptiX_INCLUDE_DIR}")
    set(OPTIX_INCLUDE_DIR "${OptiX_INCLUDE_DIR}")
endif()

# Verify that we can find optix.h
if(NOT EXISTS "${OptiX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "optix.h not found in ${OptiX_INCLUDE_DIR}. Please verify your OptiX SDK installation.")
else()
    message(STATUS "Found OptiX headers at: ${OptiX_INCLUDE_DIR}")
endif()

# Find CUDA (required for OptiX)
find_package(CUDA REQUIRED)

# ========================================
# OpenGL and Window Management
# ========================================
find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

# ========================================
# GLM
# ========================================
set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)

# ========================================
# Include Directories
# ========================================
include_directories(
    ${GLM_INCLUDE_DIRS}
    ${OptiX_INCLUDE_DIR}        # OptiX headers
    ${CUDA_INCLUDE_DIRS}        # CUDA headers
    ${CMAKE_SOURCE_DIR}/src     # Your source directory
)

# ========================================
# Source Files
# ========================================
set(headers
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/utilities.h
    src/bvh.h
    src/optixDenoiser.h   
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/intersections.cu
    src/interactions.cu
    src/scene.cpp
    src/utilities.cpp
    src/bvh.cpp
    src/optixDenoiser.cpp 
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

# ========================================
# Executable
# ========================================
add_executable(${CMAKE_PROJECT_NAME} 
    ${sources} 
    ${headers} 
    ${imgui_sources} 
    ${imgui_headers}
)

# ========================================
# Link Libraries
# ========================================
# OptiX 9.0 is primarily header-only, but we still link against CUDA
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${GL_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
    cuda  # CUDA driver API (required for OptiX)
)

# On Windows, we might need additional libraries
if(WIN32)
    target_link_libraries(${CMAKE_PROJECT_NAME}
        # Add Windows-specific libraries if needed
        ws2_32  # Windows sockets (sometimes needed)
    )
endif()

# ========================================
# CUDA Properties
# ========================================
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
    "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>"
)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
    "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>"
)

# ========================================
# Compile Definitions
# ========================================
# Add OptiX version definition if needed
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    OPTIX_VERSION=90000  # OptiX 9.0.0
)

# ========================================
# Set VS Startup Project
# ========================================
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

# ========================================
# Print Configuration Info
# ========================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "OptiX Configuration:")
message(STATUS "  OptiX Root: ${OptiX_ROOT_DIR}")
message(STATUS "  OptiX Include: ${OptiX_INCLUDE_DIR}")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "")
message(STATUS "Project Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "========================================")
message(STATUS "")

# ========================================
# Post-build Commands (Optional)
# ========================================
# If you have any OptiX samples or data files you want to copy, add them here
# For example, if you want to copy OptiX sample kernels for reference:
# add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#     "${OptiX_ROOT_DIR}/SDK/cuda"
#     "$<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>/optix_kernels"
# )