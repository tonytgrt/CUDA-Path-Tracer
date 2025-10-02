# SPDX-FileCopyrightText: Copyright (c) 2010 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
IF (WIN32 AND MSVC_VERSION LESS 1700)
  # Starting with Windows 8, the DirectX SDK is included as part of the Windows SDK
  # (http://msdn.microsoft.com/en-us/library/windows/desktop/ee663275.aspx)
  # and, in turn, Visual Studio 2012 (even Express) includes the appropriate components of the Windows SDK
  # (http://msdn.microsoft.com/en-us/windows/desktop/hh852363.aspx)
  # so we don't need a DX include path if we're targeting VS2012+
  
  FIND_PATH(DX9_INCLUDE_PATH d3d9.h 
    HINTS
    "$ENV{DXSDK_DIR}/Include"
    "$ENV{PROGRAMFILES}/Microsoft DirectX SDK/Include" 
    DOC "The directory where d3d9.h resides")
  FIND_PATH(DX10_INCLUDE_PATH D3D10.h
    HINTS 
    "$ENV{DXSDK_DIR}/Include" 
    "$ENV{PROGRAMFILES}/Microsoft DirectX SDK/Include" 
    DOC "The directory where D3D10.h resides") 
  FIND_PATH(DX11_INCLUDE_PATH D3D11.h
    HINTS
    "$ENV{DXSDK_DIR}/Include" 
    "$ENV{PROGRAMFILES}/Microsoft DirectX SDK/Include" 
    DOC "The directory where D3D11.h resides")

  IF (DX9_INCLUDE_PATH)
    SET( DX9_FOUND 1 ) 
  ELSE (DX9_INCLUDE_PATH) 
    SET( DX9_FOUND 0 ) 
  ENDIF (DX9_INCLUDE_PATH)

  IF (DX10_INCLUDE_PATH)
    SET( DX10_FOUND 1 )
  ELSE (DX10_INCLUDE_PATH) 
    SET( DX10_FOUND 0 ) 
  ENDIF (DX10_INCLUDE_PATH)  

  IF (DX11_INCLUDE_PATH)
    SET( DX11_FOUND 1 ) 
  ELSE (DX11_INCLUDE_PATH) 
    SET( DX11_FOUND 0 ) 
  ENDIF (DX11_INCLUDE_PATH)
ENDIF (WIN32 AND MSVC_VERSION LESS 1700) 

