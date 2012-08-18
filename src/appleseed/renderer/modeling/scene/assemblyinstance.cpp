
//
// This source file is part of appleseed.
// Visit http://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2010-2012 Francois Beaune, Jupiter Jazz Limited
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

// Interface header.
#include "assemblyinstance.h"

// appleseed.renderer headers.
#include "renderer/modeling/scene/assembly.h"
#include "renderer/modeling/scene/containers.h"
#include "renderer/modeling/scene/objectinstance.h"
#include "renderer/utility/bbox.h"

// Standard headers.
#include <cstddef>

using namespace foundation;
using namespace std;

namespace renderer
{

//
// AssemblyInstance class implementation.
//

namespace
{
    const UniqueID g_class_uid = new_guid();
}

AssemblyInstance::AssemblyInstance(
    const char*         name,
    const ParamArray&   params,
    const Assembly&     assembly)
  : Entity(g_class_uid, params)
  , m_assembly(assembly)
{
    set_name(name);
}

void AssemblyInstance::release()
{
    delete this;
}

bool AssemblyInstance::on_frame_begin(const Project& project)
{
    if (!m_transform_sequence.prepare())
    {
        RENDERER_LOG_ERROR("assembly instance \"%s\" has one or more invalid transforms.", get_name());
        return false;
    }

    return true;
}

void AssemblyInstance::on_frame_end(const Project& project)
{
}

GAABB3 AssemblyInstance::compute_parent_bbox() const
{
    const GAABB3 object_instances_bbox =
        get_parent_bbox<GAABB3>(
            m_assembly.object_instances().begin(),
            m_assembly.object_instances().end());

    if (m_transform_sequence.empty())
        return object_instances_bbox;

    GAABB3 bbox;
    bbox.invalidate();

    for (size_t i = 0; i < m_transform_sequence.size(); ++i)
    {
        double time;
        Transformd transform;
        m_transform_sequence.get_transform(i, time, transform);

        bbox.insert(transform.to_parent(object_instances_bbox));
    }

    return bbox;
}


//
// AssemblyInstanceFactory class implementation.
//

auto_release_ptr<AssemblyInstance> AssemblyInstanceFactory::create(
    const char*         name,
    const ParamArray&   params,
    const Assembly&     assembly)
{
    return
        auto_release_ptr<AssemblyInstance>(
            new AssemblyInstance(
                name,
                params,
                assembly));
}

}   // namespace renderer
