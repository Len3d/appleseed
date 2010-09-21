
//
// This source file is part of appleseed.
// Visit http://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2010 Francois Beaune
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

// appleseed.foundation headers.
#include "foundation/math/rng.h"
#include "foundation/math/vector.h"
#include "foundation/math/voxelgrid.h"
#include "foundation/utility/benchmark.h"

// Standard headers.
#include <cstddef>

using namespace foundation;

FOUNDATION_BENCHMARK_SUITE(Foundation_Math_VoxelGrid3)
{
    struct Fixture
    {
        static const size_t ChannelCount = 4;
        static const size_t LookupPointCount = 64;

        VoxelGrid3<float>   m_grid;
        Vector3f            m_lookup_points[LookupPointCount];
        float               m_accumulated_values[ChannelCount];

        Fixture()
          : m_grid(32, 32, 32, ChannelCount)
        {
            MersenneTwister rng;

            for (size_t z = 0; z < m_grid.get_zres(); ++z)
            {
                for (size_t y = 0; y < m_grid.get_yres(); ++y)
                {
                    for (size_t x = 0; x < m_grid.get_xres(); ++x)
                    {
                        for (size_t c = 0; c < m_grid.get_channel_count(); ++c)
                        {
                            m_grid.voxel(x, y, z)[c] = static_cast<float>(rand_double1(rng));
                        }
                    }
                }
            }

            for (size_t i = 0; i < LookupPointCount; ++i)
            {
                Vector3f& p = m_lookup_points[i];
                p[0] = static_cast<float>(rand_double1(rng, 0.0, m_grid.get_xres()));
                p[1] = static_cast<float>(rand_double1(rng, 0.0, m_grid.get_yres()));
                p[2] = static_cast<float>(rand_double1(rng, 0.0, m_grid.get_zres()));
            }

            for (size_t i = 0; i < ChannelCount; ++i)
                m_accumulated_values[i] = 0.0f;
        }
    };

    FOUNDATION_BENCHMARK_CASE_WITH_FIXTURE(TrilinearLookup, Fixture)
    {
        for (size_t i = 0; i < LookupPointCount; ++i)
        {
            __declspec(align(16)) float values[ChannelCount];
            m_grid.trilinear_lookup(m_lookup_points[i], values);

            for (size_t j = 0; j < ChannelCount; ++j)
                m_accumulated_values[j] += values[j];
        }
    }
}