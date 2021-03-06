
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
# Python.
#--------------------------------------------------------------------------------------------------

find_package (PythonLibs)

#--------------------------------------------------------------------------------------------------
# Boost libraries.
#--------------------------------------------------------------------------------------------------

find_package (Boost 1.46 REQUIRED date_time filesystem regex system thread python)


#--------------------------------------------------------------------------------------------------
# Source files.
#--------------------------------------------------------------------------------------------------

set (sources
    module.cpp
    dict2dict.hpp
    dict2dict.cpp

    bind_assembly.cpp
    bind_bbox.cpp
    bind_bsdf.cpp
    bind_camera.cpp
    bind_color.cpp
    bind_edf.cpp
    bind_entity.cpp
    bind_environment.cpp
    bind_frame.cpp
    bind_light.cpp
    bind_master_renderer.cpp
    bind_material.cpp
    bind_matrix.cpp
    bind_object.cpp
    bind_project.cpp
    bind_quaternion.cpp
    bind_renderer_controller.cpp
    bind_scene.cpp
    bind_surface_shader.cpp
    bind_texture.cpp
    bind_transform.cpp
    bind_typed_entity_containers.hpp
    bind_utility.cpp
    bind_vector.cpp
)
list (APPEND appleseed.py_sources
    ${sources}
)
source_group ("python" FILES
    ${sources}
)


#--------------------------------------------------------------------------------------------------
# Target.
#--------------------------------------------------------------------------------------------------

add_library ( _appleseed SHARED
    ${appleseed.py_sources}
)

set_target_properties (_appleseed PROPERTIES
    PREFIX ""
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${platform}/appleseed.py/appleseed
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${platform}/appleseed.py/appleseed
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${platform}/appleseed.py/appleseed
)

#--------------------------------------------------------------------------------------------------
# Include paths.
#--------------------------------------------------------------------------------------------------

include_directories (
    .
    ../appleseed.shared
	${PYTHON_INCLUDE_DIRS}
)


#--------------------------------------------------------------------------------------------------
# Preprocessor definitions.
#--------------------------------------------------------------------------------------------------

apply_preprocessor_definitions (_appleseed)


#--------------------------------------------------------------------------------------------------
# Static libraries.
#--------------------------------------------------------------------------------------------------

link_against_platform (_appleseed)

target_link_libraries (_appleseed
    appleseed
    appleseed.shared
    ${Boost_LIBRARIES}
    ${Python_LIBRARIES}
)


#--------------------------------------------------------------------------------------------------
# Post-build commands.
#--------------------------------------------------------------------------------------------------

add_copy_target_to_sandbox_command (_appleseed)
