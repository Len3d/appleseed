
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

#ifndef APPLESEED_FOUNDATION_MATH_QBVH_BVH_TO_QBVH_CONVERTOR_H
#define APPLESEED_FOUNDATION_MATH_QBVH_BVH_TO_QBVH_CONVERTOR_H

// appleseed.foundation headers.
#include "foundation/core/concepts/noncopyable.h"
#include "foundation/utility/stopwatch.h"

// Standard headers.
#include <cassert>
#include <cstddef>

namespace foundation {
namespace qbvh {

//
// BVH to QBVH convertor.
//

template <typename BVHTree, typename Tree>
class Convertor
  : public NonCopyable
{
  public:
    // Constructor.
    Convertor();

    // Convert a tree.
    template <typename Timer>
    void convert(
		BVHTree&		bvh_tree,
        Tree&           tree);

    // Return the conversion time.
    double get_build_time() const;

  private:
	typedef typename BVHTree::NodeType BVHNodeType;
    typedef typename Tree::NodeType NodeType;
    typedef typename NodeType::AABBType AABBType;

    double m_build_time;

    // Recursively convert the tree.
    void convert_recurse(
		BVHTree&		bvh_tree,
        Tree&           tree,
		BVHNodeType&	bvh_node,
        const int		parent_index,
		const int		child_index,
		const int		depth);
};


//
// Convertor class implementation.
//

template <typename BVHTree, typename Tree>
Convertor<BVHTree, Tree>::Convertor()
  : m_build_time(0.0)
{
}

template <typename BVHTree, typename Tree>
template <typename Timer>
void Convertor<BVHTree, Tree>::convert(
	BVHTree&			bvh_tree,
    Tree&               tree)
{
    // Start stopwatch.
    Stopwatch<Timer> stopwatch;
    stopwatch.start();

    // Clear the tree.
    tree.m_nodes.clear();

    // Reserve memory for the nodes.
    const size_t node_count_guess = bvh_tree.m_nodes.size() / 2;
    tree.m_nodes.reserve(node_count_guess);

    // Create the root node of the tree.
    tree.m_nodes.push_back(NodeType());

    // Recursively convert the tree.
    convert_recurse(
		bvh_tree,
        tree,
		bvh_tree.m_nodes[0],
		-1,
		0,
        0);

    // Measure and save conversion time.
    stopwatch.measure();
    m_build_time = stopwatch.get_seconds();
}

template <typename BVHTree, typename Tree>
inline double Convertor<BVHTree, Tree>::get_build_time() const
{
    return m_build_time;
}

template <typename BVHTree, typename Tree>
void Convertor<BVHTree, Tree>::convert_recurse(
	BVHTree&			bvh_tree,
    Tree&               tree,
	BVHNodeType&		bvh_node,
	const int			parent_index,
	const int			child_index,
    const int			depth)
{
	// Create a leaf if we meet a leaf
	if (node->is_leaf())
	{
		create_temp_leaf(bvh_tree, tree, bvh_node, parent_index, child_index);
		return;
	}

	int current_node = parent_index;
	int left_child_index = child_index;
	int right_child_index = child_index + 1;

	// Create an intermediate node if the depth indicates to do so
	if (depth % 2 == 0)
	{
		current_node = create_intermediate_node(bvh_tree, tree, bvh_node, parent_index, child_index);
		left_child_index = 0;
		right_child_index = 2;
	}

    // Recurse into the left subtree.
    convert_recurse(
		bvh_tree,
        tree,
		bvh_node.get_left_node(),
		current_node,
		left_child_index,
        depth + 1);

    // Recurse into the right subtree.
    convert_recurse(
		bvh_tree,
        tree,
		bvh_node.get_right_node(),
		current_node,
		right_child_index,
        depth + 1);
}

}       // namespace qbvh
}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_MATH_QBVH_BVH_TO_QBVH_CONVERTOR_H
