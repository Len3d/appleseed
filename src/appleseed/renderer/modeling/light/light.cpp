
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
#include "light.h"

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/modeling/input/inputarray.h"
#include "renderer/modeling/input/source.h"

using namespace foundation;

namespace renderer
{

//
// Light class implementation.
//

struct Light::Impl
{
    Transformd m_transform;
};

namespace
{
    const UniqueID g_class_uid = new_guid();
}

Light::Light(
    const char*         name,
    const ParamArray&   params)
  : ConnectableEntity(g_class_uid, params)
  , impl(new Impl())
{
    set_name(name);
}

Light::~Light()
{
    delete impl;
}

void Light::set_transform(const Transformd& transform)
{
    impl->m_transform = transform;
    bump_version_id();
}

const Transformd& Light::get_transform() const
{
    return impl->m_transform;
}

bool Light::on_frame_begin(
    const Project&      project,
    const Assembly&     assembly)
{
    return true;
}

void Light::on_frame_end(
    const Project&      project,
    const Assembly&     assembly)
{
}

void Light::check_exitance_input_non_null(const char* input_name) const
{
    const Source* source = m_inputs.source(input_name);

    if (source->is_uniform())
    {
        Spectrum exitance;
        Alpha alpha;
        source->evaluate_uniform(exitance, alpha);

        if (exitance == Spectrum(0.0f))
        {
            RENDERER_LOG_WARNING(
                "light \"%s\" has a zero exitance and will slow down rendering "
                "without contributing to the lighting.",
                get_name());
        }
    }
}

}   // namespace renderer
