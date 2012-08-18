
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
#include "bsdf.h"

// appleseed.renderer headers.
#include "renderer/modeling/input/inputevaluator.h"

// appleseed.foundation headers.
#include "foundation/utility/uid.h"

using namespace foundation;

namespace renderer
{

//
// BSDF class implementation.
//

const double BSDF::DiracDelta = -1.0;

namespace
{
    const UniqueID g_class_uid = new_guid();
}

BSDF::BSDF(
    const char*         name,
    const Type          type,
    const ParamArray&   params)
  : ConnectableEntity(g_class_uid, params)
  , m_type(type)
{
    set_name(name);
}

bool BSDF::on_frame_begin(
    const Project&      project,
    const Assembly&     assembly)
{
    return true;
}

void BSDF::on_frame_end(
    const Project&      project,
    const Assembly&     assembly)
{
}

size_t BSDF::compute_input_data_size(
    const Assembly&     assembly) const
{
    return get_inputs().compute_data_size();
}

void BSDF::evaluate_inputs(
    InputEvaluator&     input_evaluator,
    const Vector2d&     uv,
    const size_t        offset) const
{
    input_evaluator.evaluate(get_inputs(), uv, offset);
}

}   // namespace renderer
