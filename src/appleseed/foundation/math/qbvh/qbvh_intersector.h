
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

#ifndef APPLESEED_FOUNDATION_MATH_QBVH_QBVH_INTERSECTOR_H
#define APPLESEED_FOUNDATION_MATH_QBVH_QBVH_INTERSECTOR_H

// appleseed.foundation headers.
#include "foundation/core/concepts/noncopyable.h"
#include "foundation/math/intersection.h"
#include "foundation/math/ray.h"
#include "foundation/math/scalar.h"
#ifdef APPLESEED_FOUNDATION_USE_SSE
#include "foundation/math/fp.h"
#include "foundation/platform/compiler.h"
#include "foundation/platform/sse.h"
#endif

#include "foundation/math/qbvh/qbvh_node.h"

// Standard headers.
#include <cassert>
#include <cstddef>

namespace foundation {
namespace qbvh {

//
// QBVH intersector.
//
// The Visitor class must conform to the following prototype:
//
//      class Visitor
//        : public foundation::NonCopyable
//      {
//        public:
//          // Return whether QBVH traversal should continue or not.
//          // 'distance' should be set to the distance to the closest hit so far.
//          bool visit(
//              const NodeType&             node,
//              const RayType&              ray,
//              const RayInfoType&          ray_info,
//              ValueType&                  distance
//              );
//      };
//

template <
    typename Tree,
    typename Visitor,
    typename Ray,
    size_t StackSize = 64,
    size_t N = Tree::NodeType::AABBType::Dimension
>
class Intersector
  : public NonCopyable
{
  public:
    typedef typename Tree::NodeType NodeType;
    typedef typename NodeType::ValueType ValueType;
    typedef Ray RayType;
    typedef RayInfo<ValueType, NodeType::Dimension> RayInfoType;

    // Intersect a ray with a given BVH.
    void intersect(
        const Tree&             tree,
        const RayType&          ray,
        const RayInfoType&      ray_info,
        Visitor&                visitor
        ) const;

    // Intersect a ray with a given BVH with motion.
    void intersect(
        const Tree&             tree,
        const RayType&          ray,
        const RayInfoType&      ray_info,
        const ValueType         ray_time,
        Visitor&                visitor
        ) const;
};


//
// Intersector class implementation.
//

template <
    typename Tree,
    typename Visitor,
    typename Ray,
    size_t StackSize,
    size_t N
>
void Intersector<Tree, Visitor, Ray, StackSize, N>::intersect(
    const Tree&                 tree,
    const RayType&              ray,
    const RayInfoType&          ray_info,
    Visitor&                    visitor
    ) const
{
    // To keep the code clean, we don't have non-SSE implementation.
}

template <
    typename Tree,
    typename Visitor,
    typename Ray,
    size_t StackSize,
    size_t N
>
void Intersector<Tree, Visitor, Ray, StackSize, N>::intersect(
    const Tree&                 tree,
    const RayType&              ray,
    const RayInfoType&          ray_info,
    const ValueType             ray_time,
    Visitor&                    visitor
    ) const
{
    // To keep the code clean, we don't have non-SSE implementation.
}

#ifdef APPLESEED_FOUNDATION_USE_SSE

// The quad ray
class QuadRay
{
public:
    QuadVector   m_org;
    QuadVector   m_inv_dir;
    int32        m_sign[3];
};

template <
    typename Tree,
    typename Visitor,
    typename Ray,
    size_t StackSize
>
class Intersector<Tree, Visitor, Ray, StackSize, 3>
  : public NonCopyable
{
  public:
    typedef typename Tree::NodeType NodeType;
    typedef typename NodeType::ValueType ValueType;
    typedef Ray RayType;
    typedef RayInfo<ValueType, NodeType::Dimension> RayInfoType;

    // Intersect a ray with a given BVH.
    void intersect(
        const Tree&             tree,
        const RayType&          ray,
        const RayInfoType&      ray_info,
        Visitor&                visitor
        ) const;

    // Intersect a ray with a given BVH with motion.
    void intersect(
        const Tree&             tree,
        const RayType&          ray,
        const RayInfoType&      ray_info,
        const ValueType         ray_time,
        Visitor&                visitor
        ) const;
};

#define QMASK0 0x1
#define QMASK1 0x2
#define QMASK2 0x4
#define QMASK3 0x8

template <
    typename Tree,
    typename Visitor,
    typename Ray,
    size_t StackSize
>
void Intersector<Tree, Visitor, Ray, StackSize, 3>::intersect(
    const Tree&                 tree,
    const RayType&              ray,
    const RayInfoType&          ray_info,
    Visitor&                    visitor
    ) const
{
    // Make sure the tree was built.
    assert(!tree.m_nodes.empty());

    QuadRay qray;

    // Build quad ray
    qray.m_org.x.m = _mm_set1_ps(ray.m_org.x);
    qray.m_org.y.m = _mm_set1_ps(ray.m_org.y);
    qray.m_org.z.m = _mm_set1_ps(ray.m_org.z);
    qray.m_inv_dir.x.m = _mm_set1_ps(ray.m_rcp_dir.x);
    qray.m_inv_dir.y.m = _mm_set1_ps(ray.m_rcp_dir.y);
    qray.m_inv_dir.z.m = _mm_set1_ps(ray.m_rcp_dir.z);

    // Get ray direction signs
    qray.m_sign[0] = (ray.m_dir.x < 0.0f);
    qray.m_sign[1] = (ray.m_dir.y < 0.0f);
    qray.m_sign[2] = (ray.m_dir.z < 0.0f);

    // Node stack
    int32 nodeStack[StackSize];

    // Main loop
    int32 todoNode = 0; // The index in the stack
    nodeStack[0] = 0; // Handle root node first
    ValueType ray_tmax = ray.m_tmax;

    while (todoNode >= 0)
    {
        // Leaves are identified by a negative index
        if (!NodeType::is_leaf(nodeStack[todoNode]))
        {
            const NodeType &node = tree.m_nodes[nodeStack[todoNode]];
            --todoNode;

	        __m128 tMin = _mm_set1_ps(ray.m_tmin);
	        __m128 tMax = _mm_set1_ps(ray_tmax);

	        // X coordinate
	        tMin = _mm_max_ps(tMin, _mm_mul_ps(_mm_sub_ps(node.m_bbox[qray.m_sign[0]].x.m, 
		        qray.m_org.x.m), qray.m_inv_dir.x.m));
	        tMax = _mm_min_ps(tMax, _mm_mul_ps(_mm_sub_ps(node.m_bbox[1 - qray.m_sign[0]].x.m, 
		        qray.m_org.x.m), qray.m_inv_dir.x.m));

	        // Y coordinate
	        tMin = _mm_max_ps(tMin, _mm_mul_ps(_mm_sub_ps(node.m_bbox[qray.m_sign[1]].y.m, 
		        qray.m_org.y.m), qray.m_inv_dir.y.m));
	        tMax = _mm_min_ps(tMax, _mm_mul_ps(_mm_sub_ps(node.m_bbox[1 - qray.m_sign[1]].y.m, 
		        qray.m_org.y.m), qray.m_inv_dir.y.m));

	        // Z coordinate
	        tMin = _mm_max_ps(tMin, _mm_mul_ps(_mm_sub_ps(node.m_bbox[qray.m_sign[2]].z.m, 
		        qray.m_org.z.m), qray.m_inv_dir.z.m));
	        tMax = _mm_min_ps(tMax, _mm_mul_ps(_mm_sub_ps(node.m_bbox[1 - qray.m_sign[2]].z.m, 
		        qray.m_org.z.m), qray.m_inv_dir.z.m));

	        // Get the visit flags
	        const int32 visit = _mm_movemask_ps(_mm_cmpge_ps(tMax, tMin));

            // Do axis-based sorting for traversal order
			switch ((qray.m_sign[node.get_split_axis(0)] << 2) 
				+ (qray.m_sign[node.get_split_axis(1)] << 1) 
				+ qray.m_sign[node.get_split_axis(2)])
			{
			case 0:
				{
					// Final: 0, 1, 2, 3, stack is last in first out
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
				}
				break;
			case 1:
				{
					// Final: 0, 1, 3, 2, stack is last in first out
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
				}
				break;
			case 2:
				{
					// Final: 1, 0, 2, 3, stack is last in first out
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
				}
				break;
			case 3:
				{
					// Final: 1, 0, 3, 2, stack is last in first out
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
				}
				break;
			case 4:
				{
					// Final: 2, 3, 0, 1, stack is last in first out
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
				}
				break;
			case 5:
				{
					// Final: 3, 2, 0, 1, stack is last in first out
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
				}
				break;
			case 6:
				{
					// Final: 2, 3, 1, 0, stack is last in first out
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
				}
				break;
			case 7:
				{
					// Final: 3, 2, 1, 0, stack is last in first out
					if (visit & QMASK0)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(0);
					}
					if (visit & QMASK1)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(1);
					}
					if (visit & QMASK2)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(2);
					}
					if (visit & QMASK3)
					{
						++todoNode;
						nodeStack[todoNode] = node.get_child_node_index(3);
					}
				}
				break;
			}
        }
        else
        {
            // Visit the leaf
            ValueType distance;
#ifndef NDEBUG
            distance = ValueType(-1.0);
#endif

            int32 leafData = nodeStack[todoNode];
            --todoNode;

            if (leafData == EMPTY_LEAF_NODE)
            {
                continue;
            }

            const bool proceed =
                visitor.visit(
                    tree.m_nodes[decode_item_index(leafData)],
                    ray,
                    ray_info,
                    distance
                    );
            assert(!proceed || distance >= ValueType(0.0));

            // Terminate traversal if the visitor decided so.
            if (!proceed)
            {
                break;
            }

            // Keep track of the distance to the closest intersection.
            if (ray_tmax > distance)
            {
                ray_tmax = distance;
            }
        }
    }
}

template <
    typename Tree,
    typename Visitor,
    typename Ray,
    size_t StackSize
>
void Intersector<Tree, Visitor, Ray, StackSize, 3>::intersect(
    const Tree&                 tree,
    const RayType&              ray,
    const RayInfoType&          ray_info,
    const ValueType             ray_time,
    Visitor&                    visitor
    ) const
{
    // TODO: Support motion case here.
}

#endif

}       // namespace qbvh
}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_MATH_QBVH_QBVH_INTERSECTOR_H
