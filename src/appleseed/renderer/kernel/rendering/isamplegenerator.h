
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

#ifndef APPLESEED_RENDERER_KERNEL_RENDERING_ISAMPLEGENERATOR_H
#define APPLESEED_RENDERER_KERNEL_RENDERING_ISAMPLEGENERATOR_H

// appleseed.foundation headers.
#include "foundation/core/concepts/iunknown.h"

// Standard headers.
#include <cstddef>

// Forward declarations.
namespace foundation    { class AbortSwitch; }
namespace foundation    { class StatisticsVector; }
namespace renderer      { class AccumulationFramebuffer; }

namespace renderer
{

//
// Sample generator interface.
//

class ISampleGenerator
  : public foundation::IUnknown
{
  public:
    // Reset the sample generator to its initial state.
    virtual void reset() = 0;

    // Generate a given number of samples and store them into a progressive framebuffer.
    virtual void generate_samples(
        const size_t                sample_count,
        AccumulationFramebuffer&    framebuffer,
        foundation::AbortSwitch&    abort_switch) = 0;

    // Retrieve performance statistics.
    virtual foundation::StatisticsVector get_statistics() const = 0;
};


//
// Interface of a ISampleGenerator factory that can cross DLL boundaries.
//

class ISampleGeneratorFactory
  : public foundation::IUnknown
{
  public:
    // Return a new sample generator instance.
    virtual ISampleGenerator* create(
        const size_t                generator_index,
        const size_t                generator_count) = 0;

    // Create an accumulation framebuffer that fit this sample generator.
    virtual AccumulationFramebuffer* create_accumulation_framebuffer(
        const size_t                canvas_width,
        const size_t                canvas_height) = 0;
};

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_KERNEL_RENDERING_ISAMPLEGENERATOR_H
