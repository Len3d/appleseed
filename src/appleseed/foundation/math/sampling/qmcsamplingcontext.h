
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

#ifndef APPLESEED_FOUNDATION_MATH_SAMPLING_QMCSAMPLINGCONTEXT_H
#define APPLESEED_FOUNDATION_MATH_SAMPLING_QMCSAMPLINGCONTEXT_H

// appleseed.foundation headers.
#include "foundation/math/permutation.h"
#include "foundation/math/primes.h"
#include "foundation/math/qmc.h"
#include "foundation/math/rng.h"
#include "foundation/math/vector.h"
#include "foundation/utility/test/helpers.h"

// Standard headers.
#include <cassert>
#include <cstddef>

// Unit test case declarations.
DECLARE_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, InitialStateIsCorrect);
DECLARE_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, TestAssignmentOperator);
DECLARE_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, TestSplitting);
DECLARE_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, TestDoubleSplitting);

namespace foundation
{

//
// A sampling context featuring:
//
//   - deterministic sampling based on Hammersley and Halton sequences
//   - Faure digit scrambling
//   - Cranley-Patterson rotation
//   - Monte Carlo padding
//

template <typename RNG>
class QMCSamplingContext
{
  public:
    // Random number generator.
    typedef RNG RNGType;

    typedef Vector<double, 8> VectorType;

    // Constructors.
    explicit QMCSamplingContext(RNG& rng);
    QMCSamplingContext(
        RNG&            rng,
        const size_t    dimension,
        const size_t    sample_count,
        const size_t    instance = 0);

    // Assignment operator.
    QMCSamplingContext& operator=(const QMCSamplingContext& rhs);

    // Trajectory splitting, return a child sampling context for
    // a given number of dimensions and a given number of samples.
    QMCSamplingContext split(
        const size_t    dimension,
        const size_t    sample_count) const;

    // Set the instance number.
    void set_instance(const size_t instance);

    // Return the next sample in [0,1].
    double next_double1();

    // Return the next sample in [0,1).
    double next_double2();

    // Return the next sample in [0,1]^N.
    template <size_t N>
    Vector<double, N> next_vector1();

    // Return the next sample in [0,1)^N.
    template <size_t N>
    Vector<double, N> next_vector2();

  private:
    GRANT_ACCESS_TO_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, InitialStateIsCorrect);
    GRANT_ACCESS_TO_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, TestAssignmentOperator);
    GRANT_ACCESS_TO_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, TestSplitting);
    GRANT_ACCESS_TO_TEST_CASE(Foundation_Math_Sampling_QMCSamplingContext, TestDoubleSplitting);

    RNG&        m_rng;

    size_t      m_base_dimension;
    size_t      m_base_instance;

    size_t      m_dimension;
    size_t      m_instance;

    VectorType  m_offset;

    QMCSamplingContext(
        RNG&                rng,
        const size_t        base_dimension,
        const size_t        base_instance,
        const size_t        dimension,
        const size_t        sample_count,
        const VectorType&   offset);
};


//
// QMCSamplingContext class implementation.
//

template <typename RNG>
inline QMCSamplingContext<RNG>::QMCSamplingContext(RNG& rng)
  : m_rng(rng)
  , m_base_dimension(0)
  , m_base_instance(0)
  , m_dimension(0)
  , m_instance(0)
  , m_offset(0.0)
{
}

template <typename RNG>
inline QMCSamplingContext<RNG>::QMCSamplingContext(
    RNG&                rng,
    const size_t        dimension,
    const size_t        sample_count,
    const size_t        instance)
  : m_rng(rng)
  , m_base_dimension(0)
  , m_base_instance(0)
  , m_dimension(dimension)
  , m_instance(instance)
  , m_offset(0.0)
{
}

template <typename RNG>
inline QMCSamplingContext<RNG>::QMCSamplingContext(
    RNG&                rng,
    const size_t        base_dimension,
    const size_t        base_instance,
    const size_t        dimension,
    const size_t        sample_count,
    const VectorType&   offset)
  : m_rng(rng)
  , m_base_dimension(base_dimension)
  , m_base_instance(base_instance)
  , m_dimension(dimension)
  , m_instance(0)
  , m_offset(offset)
{
}

template <typename RNG> inline
QMCSamplingContext<RNG>&
QMCSamplingContext<RNG>::operator=(const QMCSamplingContext& rhs)
{
    m_base_dimension = rhs.m_base_dimension;
    m_base_instance = rhs.m_base_instance;
    m_dimension = rhs.m_dimension;
    m_instance = rhs.m_instance;
    m_offset = rhs.m_offset;
    return *this;
}

template <typename RNG>
inline QMCSamplingContext<RNG> QMCSamplingContext<RNG>::split(
    const size_t    dimension,
    const size_t    sample_count) const
{
    VectorType offset;

    for (size_t i = 0; i < dimension; ++i)
    {
        const size_t d = m_base_dimension + m_dimension + i;

        if (d < FaurePermutationTableSize)
        {
            assert(d < PrimeTableSize);

            offset[i] =
                permuted_radical_inverse<double>(
                    Primes[d],
                    FaurePermutations[d],
                    m_base_instance + m_instance);
        }
        else
        {
            // Monte Carlo padding.
            offset[i] = rand_double2(m_rng);
        }
    }

    return
        QMCSamplingContext(
            m_rng,
            m_base_dimension + m_dimension,     // dimension allocation
            m_base_instance + m_instance,       // decorrelation by generalization
            dimension,
            sample_count,
            offset);
}

template <typename RNG>
inline void QMCSamplingContext<RNG>::set_instance(const size_t instance)
{
    m_instance = instance;
}

template <typename RNG>
inline double QMCSamplingContext<RNG>::next_double1()
{
    return next_vector1<1>()[0];
}

template <typename RNG>
inline double QMCSamplingContext<RNG>::next_double2()
{
    return next_vector2<1>()[0];
}

template <typename RNG>
template <size_t N>
inline Vector<double, N> QMCSamplingContext<RNG>::next_vector1()
{
    // todo: implement properly.
    return next_vector2<N>();
}

template <typename RNG>
template <size_t N>
inline Vector<double, N> QMCSamplingContext<RNG>::next_vector2()
{
    assert(N == m_dimension);
    assert(N <= PrimeTableSize);
    assert(N <= FaurePermutationTableSize);

    Vector<double, N> v;

    for (size_t i = 0; i < N; ++i)
    {
        v[i] =
            permuted_radical_inverse<double>(
                Primes[i],
                FaurePermutations[i],
                m_instance);

        // Cranley-Patterson rotation.
        v[i] += m_offset[i];
        if (v[i] >= 1.0)
            v[i] -= 1.0;
    }

    ++m_instance;

    return v;
}

}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_MATH_SAMPLING_QMCSAMPLINGCONTEXT_H
