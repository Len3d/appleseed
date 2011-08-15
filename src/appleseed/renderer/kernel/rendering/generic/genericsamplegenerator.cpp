
//
// This source file is part of appleseed.
// Visit http://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2010-2011 Francois Beaune
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
#include "genericsamplegenerator.h"

// appleseed.renderer headers.
#include "renderer/global/globaltypes.h"
#include "renderer/kernel/rendering/isamplerenderer.h"
#include "renderer/kernel/rendering/localaccumulationframebuffer.h"
#include "renderer/kernel/rendering/sample.h"
#include "renderer/kernel/rendering/samplegeneratorbase.h"
#include "renderer/kernel/shading/shadingresult.h"
#include "renderer/modeling/frame/frame.h"

// appleseed.foundation headers.
#include "foundation/math/population.h"
#include "foundation/math/qmc.h"
#include "foundation/math/rng.h"
#include "foundation/math/vector.h"
#include "foundation/utility/autoreleaseptr.h"
#include "foundation/utility/string.h"

// Forward declarations.
namespace foundation    { class LightingConditions; }

using namespace foundation;

namespace renderer
{

namespace
{
    //
    // GenericSampleGenerator class implementation.
    //

    class GenericSampleGenerator
      : public SampleGeneratorBase
    {
      public:
        GenericSampleGenerator(
            const Frame&                    frame,
            ISampleRendererFactory*         sample_renderer_factory,
            const size_t                    generator_index,
            const size_t                    generator_count)
          : SampleGeneratorBase(generator_index, generator_count)
          , m_frame(frame)
          , m_sample_renderer(sample_renderer_factory->create())
          , m_lighting_conditions(frame.get_lighting_conditions())
        {
        }

        ~GenericSampleGenerator()
        {
            RENDERER_LOG_DEBUG(
                "generic sample generator statistics:\n"
                "  max. samp. dim.  avg %.1f  min %s  max %s  dev %.1f\n",
                m_max_sampling_dim.get_avg(),
                pretty_uint(m_max_sampling_dim.get_min()).c_str(),
                pretty_uint(m_max_sampling_dim.get_max()).c_str(),
                m_max_sampling_dim.get_dev());
        }

        virtual void release()
        {
            delete this;
        }

        virtual void reset()
        {
            SampleGeneratorBase::reset();
            m_rng = MersenneTwister();
        }

      private:
        const Frame&                        m_frame;
        auto_release_ptr<ISampleRenderer>   m_sample_renderer;
        const LightingConditions&           m_lighting_conditions;
        MersenneTwister                     m_rng;
        Population<size_t>                  m_max_sampling_dim;

        virtual size_t generate_samples(
            const size_t                    sequence_index,
            SampleVector&                   samples)
        {
            // Compute the sample position in NDC.
            const size_t Bases[2] = { 2, 3 };
            const Vector2d sample_position =
                halton_sequence<double, 2>(Bases, sequence_index);

            // Create a sampling context.
            SamplingContext sampling_context(
                m_rng,
                2,                          // number of dimensions
                0,                          // number of samples
                sequence_index);            // initial instance number

            // Render the sample.
            ShadingResult shading_result;
            m_sample_renderer->render_sample(
                sampling_context,
                sample_position,
                shading_result);

            // Transform the sample to the linear RGB color space.
            shading_result.transform_to_linear_rgb(m_lighting_conditions);

            // Create a single sample.
            Sample sample;
            sample.m_position = sample_position;
            sample.m_color[0] = shading_result.m_color[0];
            sample.m_color[1] = shading_result.m_color[1];
            sample.m_color[2] = shading_result.m_color[2];
            sample.m_color[3] = shading_result.m_alpha[0];
            samples.push_back(sample);

            m_max_sampling_dim.insert(sampling_context.get_max_dimension());

            return 1;
        }
    };
}


//
// GenericSampleGeneratorFactory class implementation.
//

GenericSampleGeneratorFactory::GenericSampleGeneratorFactory(
    const Frame&            frame,
    ISampleRendererFactory* sample_renderer_factory)
  : m_frame(frame)
  , m_sample_renderer_factory(sample_renderer_factory)
{
}

void GenericSampleGeneratorFactory::release()
{
    delete this;
}

ISampleGenerator* GenericSampleGeneratorFactory::create(
    const size_t            generator_index,
    const size_t            generator_count)
{
    return
        new GenericSampleGenerator(
            m_frame,
            m_sample_renderer_factory,
            generator_index,
            generator_count);
}

AccumulationFramebuffer* GenericSampleGeneratorFactory::create_accumulation_framebuffer(
    const size_t            canvas_width,
    const size_t            canvas_height)
{
    return
        new LocalAccumulationFramebuffer(
            canvas_width,
            canvas_height);
}

}   // namespace renderer
