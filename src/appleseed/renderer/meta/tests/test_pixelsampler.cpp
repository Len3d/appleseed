
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

// appleseed.renderer headers.
#include "renderer/kernel/rendering/generic/pixelsampler.h"

// appleseed.foundation headers.
#include "foundation/math/vector.h"
#include "foundation/utility/test.h"

// Standard headers.
#include <cstddef>

using namespace foundation;
using namespace renderer;
using namespace std;

TEST_SUITE(Rendering_Generic_PixelSampler)
{
    struct Fixture
    {
        PixelSampler m_sampler;

        Fixture()
        {
            // 2x2 samples per pixels.
            m_sampler.initialize(2);
        }
    };

    TEST_CASE_WITH_FIXTURE(Sample_SubPixelAtTopLeftCorner, Fixture)
    {
        Vector2d sample_position;
        size_t initial_instance;
        m_sampler.sample(0, 0, sample_position, initial_instance);

        EXPECT_EQ(Vector2d(0.0), sample_position);
    }

    TEST_CASE_WITH_FIXTURE(Sample_SubPixelAtBottomRightCorner, Fixture)
    {
        // Assume a 32x32 pixels image with 2x2 samples per pixels.
        Vector2d sample_position;
        size_t initial_instance;
        m_sampler.sample(32 * 2 - 1, 32 * 2 - 1, sample_position, initial_instance);

        EXPECT_EQ(Vector2d(31.9921875), sample_position);
    }
}