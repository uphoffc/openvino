# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(tinytc REQUIRED)
find_package(tinytc_cl REQUIRED)

set(TARGET_NAME "openvino_intel_gpu_ocl_obj")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
)
target_link_libraries(${TARGET_NAME} PRIVATE tinytc::tinytc tinytc::tinytc_cl)

ov_build_target_faster(${TARGET_NAME} PCH)
