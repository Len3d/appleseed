
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

#ifndef APPLESEED_FOUNDATION_MATH_INTERSECTION_RAYTRIANGLEMT_H
#define APPLESEED_FOUNDATION_MATH_INTERSECTION_RAYTRIANGLEMT_H

// appleseed.foundation headers.
#include "foundation/math/ray.h"
#include "foundation/math/vector.h"
#include "foundation/platform/compiler.h"

namespace foundation
{

//
// Moeller-Trumbore 3D ray-triangle intersection test.
//
// Reference:
//
//   http://jgt.akpeters.com/papers/MollerTrumbore97/
//

template <typename T>
struct TriangleMT
{
    // Types.
    typedef T ValueType;
    typedef Vector<T, 3> VectorType;
    typedef Ray<T, 3> RayType;

    // First vertex.
    VectorType  m_v0;

    // Two edges.
    VectorType  m_e0;
    VectorType  m_e1;

    // Constructors.
    TriangleMT();
    TriangleMT(
        const VectorType&   v0,
        const VectorType&   v1,
        const VectorType&   v2);

    // Construct a triangle from another triangle of a different type.
    template <typename U>
    explicit TriangleMT(const TriangleMT<U>& rhs);

    bool intersect(
        const RayType&      ray,
        ValueType&          t,
        ValueType&          u,
        ValueType&          v) const;

    bool intersect(const RayType& ray) const;
};

template <typename T>
struct TriangleMTSupportPlane
{
    // Types.
    typedef T ValueType;
    typedef Vector<T, 3> VectorType;
    typedef Ray<T, 3> RayType;

    // Triangle plane.
    VectorType  m_n;
    ValueType   m_d;

    // Constructors.
    TriangleMTSupportPlane();
    explicit TriangleMTSupportPlane(const TriangleMT<T>& triangle);

    void initialize(const TriangleMT<T>& triangle);

    ValueType intersect(
        const VectorType&   org,
        const VectorType&   dir) const;
};


//
// TriangleMT class implementation.
//

template <typename T>
inline TriangleMT<T>::TriangleMT()
{
}

template <typename T>
inline TriangleMT<T>::TriangleMT(
    const VectorType&       v0,
    const VectorType&       v1,
    const VectorType&       v2)
  : m_v0(v0)
  , m_e0(v1 - v0)
  , m_e1(v2 - v0)
{
}

template <typename T>
template <typename U>
FORCE_INLINE TriangleMT<T>::TriangleMT(const TriangleMT<U>& rhs)
  : m_v0(VectorType(rhs.m_v0))
  , m_e0(VectorType(rhs.m_e0))
  , m_e1(VectorType(rhs.m_e1))
{
}

template <typename T>
FORCE_INLINE bool TriangleMT<T>::intersect(
    const RayType&          ray,
    ValueType&              t,
    ValueType&              u,
    ValueType&              v) const
{
    // Calculate determinant.
    const VectorType pvec = cross(ray.m_dir, m_e1);
    const ValueType det = dot(m_e0, pvec);

    // Calculate distance from v0 to ray origin.
    const VectorType tvec = ray.m_org - m_v0;

    VectorType qvec;
    if (det > ValueType(0.0))
    {
        // Calculate u parameter and test bounds.
        u = dot(tvec, pvec);
        if (u < ValueType(0.0) || u > det)
            return false;

        // Prepare to test v parameter.
        qvec = cross(tvec, m_e0);

        // Calculate v parameter and test bounds.
        v = dot(ray.m_dir, qvec);
        if (v < ValueType(0.0) || u + v > det)
            return false;

        // Calculate t parameter and test bounds.
        t = dot(m_e1, qvec);
        if (t >= ray.m_tmax * det || t < ray.m_tmin * det)
            return false;
    }
    else
    {
        // Calculate u parameter and test bounds.
        u = dot(tvec, pvec);
        if (u > ValueType(0.0) || u < det)
            return false;

        // Prepare to test v parameter.
        qvec = cross(tvec, m_e0);

        // Calculate v parameter and test bounds.
        v = dot(ray.m_dir, qvec);
        if (v > ValueType(0.0) || u + v < det)
            return false;

        // Calculate t parameter and test bounds.
        t = dot(m_e1, qvec);
        if (t <= ray.m_tmax * det || t > ray.m_tmin * det)
            return false;
    }

    // Scale parameters.
    const ValueType rcp_det = ValueType(1.0) / det;
    t *= rcp_det;
    u *= rcp_det;
    v *= rcp_det;

    // Ray intersects triangle.
    return true;
}

template <typename T>
FORCE_INLINE bool TriangleMT<T>::intersect(const RayType& ray) const
{
    // Calculate determinant.
    const VectorType pvec = cross(ray.m_dir, m_e1);
    const ValueType det = dot(m_e0, pvec);

    // Calculate distance from v0 to ray origin.
    const VectorType tvec = ray.m_org - m_v0;

    VectorType qvec;
    if (det > ValueType(0.0))
    {
        // Calculate u parameter and test bounds.
        const ValueType u = dot(tvec, pvec);
        if (u < ValueType(0.0) || u > det)
            return false;

        // Prepare to test v parameter.
        qvec = cross(tvec, m_e0);

        // Calculate v parameter and test bounds.
        const ValueType v = dot(ray.m_dir, qvec);
        if (v < ValueType(0.0) || u + v > det)
            return false;

        // Calculate t parameter and test bounds.
        const ValueType t = dot(m_e1, qvec);
        if (t >= ray.m_tmax * det || t < ray.m_tmin * det)
            return false;
    }
    else
    {
        // Calculate u parameter and test bounds.
        const ValueType u = dot(tvec, pvec);
        if (u > ValueType(0.0) || u < det)
            return false;

        // Prepare to test v parameter.
        qvec = cross(tvec, m_e0);

        // Calculate v parameter and test bounds.
        const ValueType v = dot(ray.m_dir, qvec);
        if (v > ValueType(0.0) || u + v < det)
            return false;

        // Calculate t parameter and test bounds.
        const ValueType t = dot(m_e1, qvec);
        if (t <= ray.m_tmax * det || t > ray.m_tmin * det)
            return false;
    }

    // Ray intersects triangle.
    return true;
}


//
// TriangleMTSupportPlane class implementation.
//

template <typename T>
inline TriangleMTSupportPlane<T>::TriangleMTSupportPlane()
{
}

template <typename T>
inline TriangleMTSupportPlane<T>::TriangleMTSupportPlane(const TriangleMT<T>& triangle)
{
    initialize(triangle);
}

template <typename T>
inline void TriangleMTSupportPlane<T>::initialize(const TriangleMT<T>& triangle)
{
    m_n = cross(triangle.m_e0, triangle.m_e1);
    m_d = dot(triangle.m_v0, m_n);
}

template <typename T>
inline T TriangleMTSupportPlane<T>::intersect(
    const VectorType&       org,
    const VectorType&       dir) const
{
    return (m_d - dot(org, m_n)) / dot(dir, m_n);
}

}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_MATH_INTERSECTION_RAYTRIANGLEMT_H
