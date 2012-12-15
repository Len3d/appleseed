
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

#ifndef APPLESEED_FOUNDATION_MATH_QBVH_QBVH_NODE_H
#define APPLESEED_FOUNDATION_MATH_QBVH_QBVH_NODE_H

// appleseed.foundation headers.
#include "foundation/platform/compiler.h"
#include "foundation/platform/types.h"

// Standard headers.
#include <cassert>
#include <cstddef>

// SSE headers.
#include <xmmintrin.h>

namespace foundation {
namespace qbvh {

// The constant used to represent empty leaves.
#define EMPTY_LEAF_NODE		0xFFFFFFFF
#define MAX_SCALAR          std::numeric_limits<float>::max()

//
// Helper class, SSE scalar.
//
typedef struct QuadScalar {
    union {
		struct {
			float   	x;
			float   	y;
			float   	z;
			float   	w;
		};
		float   		v[4];
		__m128			m;
	};
} QuadScalar;

//
// Helper class, SSE vector.
//
typedef struct QuadVector {
	QuadScalar	    x;
	QuadScalar	    y;
	QuadScalar	    z;
} QuadVector;

//
// Node (leaf node or interior node) of a QBVH.
// User data is not supported by QBVH nodes, because 
// we directly encode leave data in its parent.
//

template <typename AABB>
class FOUNDATION_ALIGN(64) Node
{
  public:
    typedef AABB AABBType;

    // Constructor.
    Node();

    // Get the node type by node index.
    static bool is_leaf(const int32 node_index);
    // Get the index of the first item by node index.
    static size_t decode_item_index(const int32 node_index);

    // Set/get the split axis of the i-th child node (interior nodes only).
    void set_split_axis(const size_t i, const size_t axis);
    size_t get_split_axis(const size_t i) const;

    // Set/get the bounding boxes of the i-th child node (interior nodes only, static case).
    void set_bbox(const size_t i, const AABBType& bbox);
    AABBType get_bbox(const size_t i) const;

    // TODO: We should support motion case here.

    // Set/get the index of the i-th child node (interior nodes only).
    void set_child_node_index(const size_t i, const size_t index);
    size_t get_child_node_index(const size_t i) const;

    // Set the index of the first item of the i-th child node (leaf nodes only).
    void set_item_index(const size_t i, const size_t index);
    size_t get_item_index(const size_t i) const;

    // Set/get the item count of the i-th child node (leaf nodes only).
    void set_item_count(const size_t i, const size_t count);
    size_t get_item_count(const size_t i) const;

  private:
    template <typename Tree, typename Visitor, typename Ray, size_t StackSize, size_t N>
    friend class Intersector;

    typedef typename AABBType::ValueType ValueType;
    static const size_t Dimension = AABBType::Dimension;

    // The bounding boxes of 4 child nodes
    QuadVector               m_bbox[2];

    // The first 1 bit encode if the child is a leaf, the next 31 bits encode 
    // the first primitive index
    int32					m_child[4];
    // The first 2 bits encode the split axis, the next 30 bits encode the 
    // primitive count
    uint32					m_flags[4];
};


//
// Node class implementation.
//

template <typename AABB>
inline Node<AABB>::Node()
{
    // Set to empty bounding boxes
    m_bbox[0].x.m = _mm_set1_ps(MAX_SCALAR);
	m_bbox[0].y.m = _mm_set1_ps(MAX_SCALAR);
	m_bbox[0].z.m = _mm_set1_ps(MAX_SCALAR);
	m_bbox[1].x.m = _mm_set1_ps(-MAX_SCALAR);
	m_bbox[1].y.m = _mm_set1_ps(-MAX_SCALAR);
	m_bbox[1].z.m = _mm_set1_ps(-MAX_SCALAR);

    // All children are empty leaves by default
    m_child[0] = EMPTY_LEAF_NODE;
    m_child[1] = EMPTY_LEAF_NODE;
    m_child[2] = EMPTY_LEAF_NODE;
    m_child[3] = EMPTY_LEAF_NODE;
    m_flags[0] = 0;
    m_flags[1] = 0;
    m_flags[2] = 0;
    m_flags[3] = 0;
}

template <typename AABB>
inline bool Node<AABB>::is_leaf(const int32 node_index)
{
    return (node_index < 0);
}

template <typename AABB>
inline size_t Node<AABB>::decode_item_index(const int32 node_index)
{
    return static_cast<size_t>(node_index & 0x7FFFFFFF);
}

template <typename AABB>
inline void Node<AABB>::set_split_axis(const size_t i, const size_t axis)
{
    m_flags[i] &= 0x3FFFFFFF; // Clear the split axis bits
	m_flags[i] |= ((static_cast<int32>(axis) & 3) << 30);
}

template <typename AABB>
inline size_t Node<AABB>::get_split_axis(const size_t i) const
{
    return static_cast<size_t>(m_flags[i] >> 30);
}

template <typename AABB>
inline void Node<AABB>::set_bbox(const size_t i, const AABBType& bbox)
{
    m_bbox[0].x.v[i] = bbox.min[0];
	m_bbox[0].y.v[i] = bbox.min[1];
	m_bbox[0].z.v[i] = bbox.min[2];
	m_bbox[1].x.v[i] = bbox.max[0];
	m_bbox[1].y.v[i] = bbox.max[1];
	m_bbox[1].z.v[i] = bbox.max[2];
}

template <typename AABB>
inline AABB Node<AABB>::get_bbox(const size_t i) const
{
    AABBType bbox;

    bbox.min[0] = m_bbox[0].x.v[i];
	bbox.min[1] = m_bbox[0].y.v[i];
	bbox.min[2] = m_bbox[0].z.v[i];
	bbox.max[0] = m_bbox[1].x.v[i];
	bbox.max[1] = m_bbox[1].y.v[i];
	bbox.max[2] = m_bbox[1].z.v[i];

    return bbox;
}

template <typename AABB>
inline void Node<AABB>::set_child_node_index(const size_t i, const size_t index)
{
    // Currently we have limited bits of integer
    assert(index <= 0xFFFFFFFFUL);
    
    m_child[i] = static_cast<int32>(index);
}

template <typename AABB>
inline size_t Node<AABB>::get_child_node_index(const size_t i) const
{
    return static_cast<size_t>(m_child[i]);
}

template <typename AABB>
inline void Node<AABB>::set_item_index(const size_t i, const size_t index)
{
    // Currently we have limited bits of integer
    assert(index <= 0x7FFFFFFFUL);

    m_child[i] |= (static_cast<int32>(index) & 0x7FFFFFFF);
}

template <typename AABB>
inline size_t Node<AABB>::get_item_index(const size_t i) const
{
    return decode_item_index(m_child[i]);
}

template <typename AABB>
inline void Node<AABB>::set_item_count(const size_t i, const size_t count)
{
    m_flags[i] &= (3 << 30); // Clear the count bits
    m_flags[i] |= (static_cast<uint32>(count) & 0x3FFFFFFF);
}

template <typename AABB>
inline size_t Node<AABB>::get_item_count(const size_t i) const
{
    return static_cast<size_t>(m_flags[i] & 0x3FFFFFFF);
}

}       // namespace qbvh
}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_MATH_QBVH_QBVH_NODE_H
