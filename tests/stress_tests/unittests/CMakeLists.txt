# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "StressUnitTests")

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE StressTestsCommon)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
