# SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This is a wrapper for FindZLIB that returns the static library instead of the DSO/DLL.

# Optional input variable: ZlibStatic_ROOT
# Output variables:
#   ZlibStatic_FOUND
#   ZlibStatic_INCLUDE_DIR
#   ZlibStatic_LIBRARIES
#   ZlibStatic_VERSION

# FindZLIB honors ZLIB_ROOT, but the lack of a pre-existing cache entry for it is not user-friendly.
set( ZlibStatic_ROOT "" CACHE PATH "Path to Zlib installation directory" )
if( ZlibStatic_ROOT AND NOT ZLIB_ROOT )
  set( ZLIB_ROOT "${ZlibStatic_ROOT}" CACHE PATH "Path to Zlib installation directory" FORCE )
  unset( ZLIB_INCLUDE_DIR CACHE )
  unset( ZLIB_LIBRARY_RELEASE CACHE )
  unset( ZLIB_LIBRARY_DEBUG CACHE )
endif()

find_package( ZLIB )
if( NOT ZLIB_FOUND OR NOT ZLIB_LIBRARY_RELEASE )
  return()
endif()

# Verify that zlibstatic exists alongside the zlib library.
get_filename_component( LIB_DIR ${ZLIB_LIBRARY_RELEASE} DIRECTORY )

get_filename_component( LIB_FILE_RELEASE ${ZLIB_LIBRARY_RELEASE} NAME )
string( REGEX REPLACE "zlib" "zlibstatic" LIB_FILE_RELEASE "${LIB_FILE_RELEASE}" )
file( GLOB ZlibStatic_LIBRARY_RELEASE "${LIB_DIR}/${LIB_FILE_RELEASE}" )

if( ZLIB_LIBRARY_DEBUG )
  get_filename_component( LIB_FILE_DEBUG ${ZLIB_LIBRARY_DEBUG} NAME )
  string( REGEX REPLACE "zlib" "zlibstatic" LIB_FILE_DEBUG "${LIB_FILE_DEBUG}" )
  file( GLOB ZlibStatic_LIBRARY_DEBUG "${LIB_DIR}/${LIB_FILE_DEBUG}" )
else()
  # Fall back on release library if debug library is not found.
  set( ZlibStatic_LIBRARY_DEBUG "${ZlibStatic_LIBRARY_RELEASE}"
    CACHE FILEPATH "Path to debug Zlib library" )
endif()

if ( ZlibStatic_LIBRARY_RELEASE AND ZlibStatic_LIBRARY_DEBUG )
  set( ZlibStatic_LIBRARIES "optimized;${ZlibStatic_LIBRARY_RELEASE};debug;${ZlibStatic_LIBRARY_DEBUG}"
    CACHE STRING "Zlib static libraries" )
endif()
set( ZlibStatic_INCLUDE_DIR "${ZLIB_INCLUDE_DIR}"
  CACHE PATH "Path to Zlib include directory" )
set( ZlibStatic_VERSION "${ZLIB_VERSION_STRING}"
  CACHE STRING "Zlib version number" )

find_package_handle_standard_args( ZlibStatic
  REQUIRED_VARS
  ZlibStatic_LIBRARY_RELEASE
  ZlibStatic_INCLUDE_DIR
  VERSION_VAR ZlibStatic_VERSION )

if( ZlibStatic_FOUND )
    add_library( Zlib::Static STATIC IMPORTED )
    set_target_properties( Zlib::Static PROPERTIES 
        # Use the release configuration by default
        IMPORTED_LOCATION ${ZlibStatic_LIBRARY_RELEASE}
        IMPORTED_LOCATION_DEBUG ${ZlibStatic_LIBRARY_DEBUG} )
    set_property( TARGET Zlib::Static APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIR} )
endif()
