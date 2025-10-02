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

IF(EXISTS "/proc/cpuinfo")

  FILE(READ /proc/cpuinfo PROC_CPUINFO)

  SET(VENDOR_ID_RX "vendor_id[ \t]*:[ \t]*([a-zA-Z]+)\n")
  STRING(REGEX MATCH "${VENDOR_ID_RX}" VENDOR_ID "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${VENDOR_ID_RX}" "\\1" VENDOR_ID "${VENDOR_ID}")

  SET(CPU_FAMILY_RX "cpu family[ \t]*:[ \t]*([0-9]+)")
  STRING(REGEX MATCH "${CPU_FAMILY_RX}" CPU_FAMILY "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${CPU_FAMILY_RX}" "\\1" CPU_FAMILY "${CPU_FAMILY}")

  SET(MODEL_RX "model[ \t]*:[ \t]*([0-9]+)")
  STRING(REGEX MATCH "${MODEL_RX}" MODEL "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${MODEL_RX}" "\\1" MODEL "${MODEL}")

  SET(FLAGS_RX "flags[ \t]*:[ \t]*([a-zA-Z0-9 _]+)\n")
  STRING(REGEX MATCH "${FLAGS_RX}" FLAGS "${PROC_CPUINFO}")
  STRING(REGEX REPLACE "${FLAGS_RX}" "\\1" FLAGS "${FLAGS}")

  # Debug output.
  IF(LINUX_CPUINFO)
    MESSAGE(STATUS "LinuxCPUInfo.cmake:")
    MESSAGE(STATUS "VENDOR_ID : ${VENDOR_ID}")
    MESSAGE(STATUS "CPU_FAMILY : ${CPU_FAMILY}")
    MESSAGE(STATUS "MODEL : ${MODEL}")
    MESSAGE(STATUS "FLAGS : ${FLAGS}")
  ENDIF(LINUX_CPUINFO)

ENDIF(EXISTS "/proc/cpuinfo")
