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

# This script produces a string variable from the contents of a ptx
# script.  The variable is defined in the .cc file and the .h file.

# This script excepts the following variable to be passed in like
# -DVAR:TYPE=VALUE
# 
# CPP_FILE
# PTX_FILE
# VARIABLE_NAME
# NAMESPACE
# CUDA_BIN2C_EXECUTABLE

# message("PTX_FILE      = ${PTX_FILE}")
# message("CPP_FILE      = ${C_FILE}")
# message("VARIABLE_NAME = ${VARIABLE_NAME}")
# message("NAMESPACE     = ${NAMESPACE}")

execute_process( COMMAND ${CUDA_BIN2C_EXECUTABLE} -p 0 -st -c -n ${VARIABLE_NAME}_static "${PTX_FILE}"
  OUTPUT_VARIABLE bindata
  RESULT_VARIABLE result
  ERROR_VARIABLE error
  )
if(result)
  message(FATAL_ERROR "bin2c error:\n" ${error})
endif()

set(BODY
  "${bindata}\n"
  "namespace ${NAMESPACE} {\n\nstatic const char* const ${VARIABLE_NAME} = reinterpret_cast<const char*>(&${VARIABLE_NAME}_static[0]);\n} // end namespace ${NAMESPACE}\n")
file(WRITE ${CPP_FILE} "${BODY}")
