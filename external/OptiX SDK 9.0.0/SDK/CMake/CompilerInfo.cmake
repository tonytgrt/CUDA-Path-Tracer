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

# Sets some variables depending on which compiler you are using
#
# USING_GNU_C       : gcc is being used for C compiler
# USING_GNU_CXX     : g++ is being used for C++ compiler
# USING_CLANG_C     : gcc is being used for C compiler
# USING_CLANG_CXX   : g++ is being used for C++ compiler
# USING_ICC         : icc is being used for C compiler
# USING_ICPC        : icpc is being used for C++ compiler
# USING_WINDOWS_CL  : Visual Studio's compiler
# USING_WINDOWS_ICL : Intel's Windows compiler

set(USING_KNOWN_C_COMPILER TRUE)
if(CMAKE_COMPILER_IS_GNUCC)
  set(USING_GNU_C TRUE)
elseif( CMAKE_C_COMPILER_ID STREQUAL "Intel" )
  set(USING_ICC TRUE)
elseif( CMAKE_C_COMPILER_ID STREQUAL "Clang" )
  set(USING_CLANG_C TRUE)
elseif( MSVC OR "x${CMAKE_C_COMPILER_ID}" STREQUAL "xMSVC" )
  set(USING_WINDOWS_CL TRUE)
else()
  set(USING_KNOWN_C_COMPILER FALSE)
endif()


set(USING_KNOWN_CXX_COMPILER TRUE)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(USING_GNU_CXX TRUE)
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
  set(USING_ICPC TRUE)
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  set(USING_CLANG_CXX TRUE)
elseif( MSVC OR "x${CMAKE_C_COMPILER_ID}" STREQUAL "xMSVC" )
  if( NOT USING_WINDOWS_CL )
    message( WARNING "Mixing WinCL C++ compiler with non-matching C compiler" )
  endif()
else()
  set(USING_KNOWN_CXX_COMPILER FALSE)
endif()

if(USING_GNU_C)
  execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
    OUTPUT_VARIABLE GCC_VERSION)
endif()

# Using unknown compilers
if(NOT USING_KNOWN_C_COMPILER)
  FIRST_TIME_MESSAGE("Specified C compiler ${CMAKE_C_COMPILER} is not recognized (gcc, icc).  Using CMake defaults.")
endif()

if(NOT USING_KNOWN_CXX_COMPILER)
  FIRST_TIME_MESSAGE("Specified CXX compiler ${CMAKE_CXX_COMPILER} is not recognized (g++, icpc).  Using CMake defaults.")
endif()


if(USING_WINDOWS_CL)
  # We should set this macro as well to get our nice trig functions
  add_definitions(-D_USE_MATH_DEFINES)
  # Microsoft does some stupid things like #define min and max.
  add_definitions(-DNOMINMAX)
endif()
