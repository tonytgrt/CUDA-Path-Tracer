# SPDX-FileCopyrightText: Copyright (c) 2008 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# This script copies one of two supplied dlls into the build directory based on the build configuration.

# build_configuration - Should be passed in via:

#   if(CMAKE_GENERATOR MATCHES "Visual Studio")
#     set( build_configuration "$(ConfigurationName)" )
#   else()
#     set( build_configuration "${CMAKE_BUILD_TYPE}")
#   endif()
#
#  -D build_configuration:STRING=${build_configuration}

# output_directory - should be passed in via the following.  If not supplied the current output directory is used.
#
#  -D "output_directory:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}"

# source_dll - should be the release version or the single version if you don't have a debug version
#
#  -D "source_dll:FILE=${path_to_source_dll}"

# source_debug_dll - should be the debug version of the dll (optional)
#
#  -D "source_debug_dll:FILE=${path_to_source_debug_dll}"

if(NOT DEFINED build_configuration)
  message(FATAL_ERROR "build_configuration not specified")
endif()

if(NOT DEFINED output_directory)
  set(output_directory ".")
endif()

if(NOT DEFINED source_dll)
  message(FATAL_ERROR "source_dll not specified")
endif()

if(NOT DEFINED source_debug_dll)
  set(source_debug_dll "${source_dll}")
endif()

# Compute the file name
if(build_configuration STREQUAL Debug)
  set(source "${source_debug_dll}")
else()
  set(source "${source_dll}")
endif()

get_filename_component(filename "${source}" NAME)
set(dest "${output_directory}/${filename}")
message(STATUS "Copying if different ${source} to ${dest}")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${source}" "${dest}"
  RESULT_VARIABLE result
  )
if(result)
  message(FATAL_ERROR "Error copying dll")
endif()

