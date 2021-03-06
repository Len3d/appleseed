
#
# This source file is part of appleseed.
# Visit http://appleseedhq.net/ for additional information and resources.
#
# This software is released under the MIT license.
#
# Copyright (c) 2010-2012 Francois Beaune, Jupiter Jazz Limited
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#


#--------------------------------------------------------------------------------------------------
# Source files.
#--------------------------------------------------------------------------------------------------

set (application_meta_tests_sources
    application/meta/tests/test_superlogger.cpp
)
list (APPEND appleseed.shared_sources
    ${application_meta_tests_sources}
)
source_group ("application\\meta\\tests" FILES
    ${application_meta_tests_sources}
)

set (application_sources
    application/application.cpp
    application/application.h
    application/commandlinehandlerbase.cpp
    application/commandlinehandlerbase.h
    application/superlogger.cpp
    application/superlogger.h
)
list (APPEND appleseed.shared_sources
    ${application_sources}
)
source_group ("application" FILES
    ${application_sources}
)


#--------------------------------------------------------------------------------------------------
# Target.
#--------------------------------------------------------------------------------------------------

add_library (appleseed.shared SHARED
    ${appleseed.shared_sources}
)

set_target_properties (appleseed.shared PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${platform}/appleseed.shared
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${platform}/appleseed.shared
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${platform}/appleseed.shared
)


#--------------------------------------------------------------------------------------------------
# Include paths.
#--------------------------------------------------------------------------------------------------

include_directories (
    .
)


#--------------------------------------------------------------------------------------------------
# Preprocessor definitions.
#--------------------------------------------------------------------------------------------------

apply_preprocessor_definitions (appleseed.shared)

append_custom_preprocessor_definitions (appleseed.shared
    APPLESEED_SHARED_EXPORTS
)


#--------------------------------------------------------------------------------------------------
# Static libraries.
#--------------------------------------------------------------------------------------------------

target_link_libraries (appleseed.shared
    appleseed
    ${Boost_LIBRARIES}
)


#--------------------------------------------------------------------------------------------------
# Post-build commands.
#--------------------------------------------------------------------------------------------------

add_copy_target_to_sandbox_command (appleseed.shared)
