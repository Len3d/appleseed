
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
#include "triangletree.h"

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/kernel/intersection/triangleencoder.h"
#include "renderer/kernel/intersection/triangleitemhandler.h"
#include "renderer/kernel/intersection/trianglevertexinfo.h"
#include "renderer/kernel/tessellation/statictessellation.h"
#include "renderer/kernel/texturing/texturecache.h"
#include "renderer/kernel/texturing/texturestore.h"
#include "renderer/modeling/input/source.h"
#include "renderer/modeling/material/material.h"
#include "renderer/modeling/object/iregion.h"
#include "renderer/modeling/object/object.h"
#include "renderer/modeling/object/regionkit.h"
#include "renderer/modeling/object/triangle.h"
#include "renderer/modeling/scene/assembly.h"
#include "renderer/modeling/scene/containers.h"
#include "renderer/modeling/scene/objectinstance.h"
#include "renderer/utility/bbox.h"
#include "renderer/utility/paramarray.h"

// appleseed.foundation headers.
#include "foundation/math/area.h"
#include "foundation/math/permutation.h"
#include "foundation/math/treeoptimizer.h"
#include "foundation/platform/system.h"
#include "foundation/platform/timer.h"
#include "foundation/utility/memory.h"
#include "foundation/utility/statistics.h"

// Standard headers.
#include <algorithm>
#include <cassert>

using namespace foundation;
using namespace std;

namespace renderer
{

//
// TriangleTree class implementation.
//

namespace
{
    template <typename Vector>
    void increase_capacity(Vector* vec, const size_t count)
    {
        if (vec)
            vec->reserve(vec->capacity() + count);
    }

    template <typename Vector>
    void assert_empty(Vector* vec)
    {
        assert(vec == 0 || vec->empty());
    }

    template <typename AABBType>
    void collect_static_triangles(
        const GAABB3&                   tree_bbox,
        const RegionInfo&               region_info,
        const StaticTriangleTess&       tess,
        const Transformd&               transform,
        vector<TriangleKey>*            triangle_keys,
        vector<TriangleVertexInfo>*     triangle_vertex_infos,
        vector<GVector3>*               triangle_vertices,
        vector<AABBType>*               triangle_bboxes,
        size_t&                         triangle_vertex_count)
    {
        const size_t triangle_count = tess.m_primitives.size();

        // Reserve memory.
        increase_capacity(triangle_keys, triangle_count);
        increase_capacity(triangle_vertex_infos, triangle_count);
        increase_capacity(triangle_vertices, triangle_count * 3);
        increase_capacity(triangle_bboxes, triangle_count);

        for (size_t i = 0; i < triangle_count; ++i)
        {
            // Fetch the triangle.
            const Triangle& triangle = tess.m_primitives[i];

            // Retrieve the object space vertices of the triangle.
            const GVector3& v0_os = tess.m_vertices[triangle.m_v0];
            const GVector3& v1_os = tess.m_vertices[triangle.m_v1];
            const GVector3& v2_os = tess.m_vertices[triangle.m_v2];

            // Transform triangle vertices to assembly space.
            const GVector3 v0 = transform.point_to_parent(v0_os);
            const GVector3 v1 = transform.point_to_parent(v1_os);
            const GVector3 v2 = transform.point_to_parent(v2_os);

            // Compute the bounding box of the triangle.
            GAABB3 triangle_bbox;
            triangle_bbox.invalidate();
            triangle_bbox.insert(v0);
            triangle_bbox.insert(v1);
            triangle_bbox.insert(v2);

            // Ignore degenerate triangles.
            if (square_area(v0, v1, v2) == GScalar(0.0))
                continue;

            // Ignore triangles that don't intersect the tree.
            if (!intersect(tree_bbox, v0, v1, v2))
                continue;

            // Store the triangle key.
            if (triangle_keys)
            {
                triangle_keys->push_back(
                    TriangleKey(
                        region_info.get_object_instance_index(),
                        region_info.get_region_index(),
                        i));
            }

            // Store the index of the first triangle vertex and the number of motion segments.
            if (triangle_vertex_infos)
            {
                triangle_vertex_infos->push_back(
                    TriangleVertexInfo(triangle_vertex_count, 0));
            }

            // Store the triangle vertices.
            if (triangle_vertices)
            {
                triangle_vertices->push_back(v0);
                triangle_vertices->push_back(v1);
                triangle_vertices->push_back(v2);
            }
            triangle_vertex_count += 3;

            // Store the triangle bounding box.
            if (triangle_bboxes)
                triangle_bboxes->push_back(AABBType(triangle_bbox));
        }
    }

    template <typename AABBType>
    void collect_moving_triangles(
        const GAABB3&                   tree_bbox,
        const RegionInfo&               region_info,
        const StaticTriangleTess&       tess,
        const Transformd&               transform,
        const double                    time,
        vector<TriangleKey>*            triangle_keys,
        vector<TriangleVertexInfo>*     triangle_vertex_infos,
        vector<GVector3>*               triangle_vertices,
        vector<AABBType>*               triangle_bboxes,
        size_t&                         triangle_vertex_count)
    {
        const size_t motion_segment_count = tess.get_motion_segment_count();
        const size_t triangle_count = tess.m_primitives.size();

        // Reserve memory.
        increase_capacity(triangle_keys, triangle_count);
        increase_capacity(triangle_vertex_infos, triangle_count);
        increase_capacity(triangle_vertices, triangle_count * 3 * (motion_segment_count + 1));
        increase_capacity(triangle_bboxes, triangle_count);

        vector<GAABB3> tri_pose_bboxes(motion_segment_count + 1);

        for (size_t i = 0; i < triangle_count; ++i)
        {
            // Fetch the triangle.
            const Triangle& triangle = tess.m_primitives[i];

            // Retrieve the object space vertices of the triangle.
            const GVector3& v0_os = tess.m_vertices[triangle.m_v0];
            const GVector3& v1_os = tess.m_vertices[triangle.m_v1];
            const GVector3& v2_os = tess.m_vertices[triangle.m_v2];

            // Transform triangle vertices to assembly space.
            const GVector3 v0 = transform.point_to_parent(v0_os);
            const GVector3 v1 = transform.point_to_parent(v1_os);
            const GVector3 v2 = transform.point_to_parent(v2_os);

            // Compute the bounding box of the triangle for each of its pose.
            tri_pose_bboxes[0].invalidate();
            tri_pose_bboxes[0].insert(v0);
            tri_pose_bboxes[0].insert(v1);
            tri_pose_bboxes[0].insert(v2);
            for (size_t m = 0; m < motion_segment_count; ++m)
            {
                tri_pose_bboxes[m + 1].invalidate();
                tri_pose_bboxes[m + 1].insert(transform.point_to_parent(tess.get_vertex_pose(triangle.m_v0, m)));
                tri_pose_bboxes[m + 1].insert(transform.point_to_parent(tess.get_vertex_pose(triangle.m_v1, m)));
                tri_pose_bboxes[m + 1].insert(transform.point_to_parent(tess.get_vertex_pose(triangle.m_v2, m)));
            }

            // Compute the bounding box of the triangle over its entire motion.
            const GAABB3 triangle_motion_bbox =
                compute_union<GAABB3>(tri_pose_bboxes.begin(), tri_pose_bboxes.end());

            // Ignore triangles that are degenerate over their entire motion.
            if (triangle_motion_bbox.rank() < 2)
                continue;

            // Ignore triangles that don't ever intersect the tree.
            if (!GAABB3::overlap(tree_bbox, triangle_motion_bbox))
                continue;

            // Store the triangle key.
            if (triangle_keys)
            {
                triangle_keys->push_back(
                    TriangleKey(
                        region_info.get_object_instance_index(),
                        region_info.get_region_index(),
                        i));
            }

            // Store the index of the first triangle vertex and the number of motion segments.
            if (triangle_vertex_infos)
            {
                triangle_vertex_infos->push_back(
                    TriangleVertexInfo(
                        triangle_vertex_count,
                        motion_segment_count));
            }

            // Store the triangle vertices.
            if (triangle_vertices)
            {
                triangle_vertices->push_back(v0);
                triangle_vertices->push_back(v1);
                triangle_vertices->push_back(v2);
                for (size_t m = 0; m < motion_segment_count; ++m)
                {
                    triangle_vertices->push_back(transform.point_to_parent(tess.get_vertex_pose(triangle.m_v0, m)));
                    triangle_vertices->push_back(transform.point_to_parent(tess.get_vertex_pose(triangle.m_v1, m)));
                    triangle_vertices->push_back(transform.point_to_parent(tess.get_vertex_pose(triangle.m_v2, m)));
                }
            }
            triangle_vertex_count += (motion_segment_count + 1) * 3;

            // Compute and store the bounding box of the triangle for the time value passed in argument.
            if (triangle_bboxes)
            {
                const GAABB3 triangle_midtime_bbox =
                    interpolate<GAABB3>(tri_pose_bboxes.begin(), tri_pose_bboxes.end(), time);
                triangle_bboxes->push_back(AABBType(triangle_midtime_bbox));
            }
        }
    }

    template <typename AABBType>
    void collect_triangles(
        const TriangleTree::Arguments&  arguments,
        const double                    time,
        vector<TriangleKey>*            triangle_keys,
        vector<TriangleVertexInfo>*     triangle_vertex_infos,
        vector<GVector3>*               triangle_vertices,
        vector<AABBType>*               triangle_bboxes)
    {
        assert_empty(triangle_keys);
        assert_empty(triangle_vertex_infos);
        assert_empty(triangle_vertices);
        assert_empty(triangle_bboxes);

        size_t triangle_vertex_count = 0;

        const size_t region_count = arguments.m_regions.size();

        for (size_t i = 0; i < region_count; ++i)
        {
            // Fetch the region info.
            const RegionInfo& region_info = arguments.m_regions[i];

            // Retrieve the object instance and its transformation.
            const ObjectInstance* object_instance =
                arguments.m_assembly.object_instances().get_by_index(
                    region_info.get_object_instance_index());
            assert(object_instance);

            // Retrieve the object.
            Object& object = object_instance->get_object();

            // Retrieve the region kit of the object.
            Access<RegionKit> region_kit(&object.get_region_kit());

            // Retrieve the region.
            const IRegion* region = (*region_kit)[region_info.get_region_index()];

            // Retrieve the tessellation of the region.
            Access<StaticTriangleTess> tess(&region->get_static_triangle_tess());

            // Collect the triangles from this tessellation.
            if (tess->get_motion_segment_count() > 0)
            {
                collect_moving_triangles(
                    arguments.m_bbox,
                    region_info,
                    tess.ref(),
                    object_instance->get_transform(),
                    time,
                    triangle_keys,
                    triangle_vertex_infos,
                    triangle_vertices,
                    triangle_bboxes,
                    triangle_vertex_count);
            }
            else
            {
                collect_static_triangles(
                    arguments.m_bbox,
                    region_info,
                    tess.ref(),
                    object_instance->get_transform(),
                    triangle_keys,
                    triangle_vertex_infos,
                    triangle_vertices,
                    triangle_bboxes,
                    triangle_vertex_count);
            }
        }
    }
}

TriangleTree::Arguments::Arguments(
    const Scene&            scene,
    const UniqueID          triangle_tree_uid,
    const GAABB3&           bbox,
    const Assembly&         assembly,
    const RegionInfoVector& regions)
  : m_scene(scene)
  , m_triangle_tree_uid(triangle_tree_uid)
  , m_bbox(bbox)
  , m_assembly(assembly)
  , m_regions(regions)
{
}

TriangleTree::TriangleTree(const Arguments& arguments)
  : TreeType(AlignedAllocator<void>(System::get_l1_data_cache_line_size()))
  , m_triangle_tree_uid(arguments.m_triangle_tree_uid)
{
    Statistics statistics;

    const string DefaultAccelerationStructure = "bvh";

    string acceleration_structure =
        arguments.m_assembly.get_parameters().get_optional<string>(
            "acceleration_structure",
            DefaultAccelerationStructure);

    if (acceleration_structure != "bvh" && acceleration_structure != "sbvh")
    {
        RENDERER_LOG_DEBUG(
            "while building acceleration structure for assembly \"%s\": "
            "invalid acceleration structure \"%s\", using default value \"%s\".",
            arguments.m_assembly.get_name(),
            acceleration_structure.c_str(),
            DefaultAccelerationStructure.c_str());

        acceleration_structure = DefaultAccelerationStructure;
    }

    // We're going to build the BVH for the geometry as it is in the middle of the shutter interval:
    // the topology of the tree will be optimal for this time and will hopefully be good enough for
    // all the shutter interval.
    const double Time = 0.5;

    // Build the tree.
    if (acceleration_structure == "bvh")
        build_bvh(arguments, Time, statistics);
    else build_sbvh(arguments, Time, statistics);

#ifdef RENDERER_TRIANGLE_TREE_REORDER_NODES
    // Optimize the tree layout in memory.
    TreeOptimizer<NodeVectorType> tree_optimizer(m_nodes);
    tree_optimizer.optimize_node_layout(TriangleTreeSubtreeDepth);
    assert(m_nodes.size() == m_nodes.capacity());
#endif

    // Create intersection filters.
    if (arguments.m_assembly.get_parameters().get_optional<bool>("enable_intersection_filters", true))
        create_intersection_filters(arguments, statistics);
    m_has_intersection_filters = !m_intersection_filters.empty();

    // Print triangle tree statistics.
    statistics.insert_size("nodes alignment", alignment(&m_nodes[0]));
    RENDERER_LOG_DEBUG("%s",
        StatisticsVector::make(
            "triangle tree #" + to_string(arguments.m_triangle_tree_uid) + " statistics",
            statistics).to_string().c_str());
}

TriangleTree::~TriangleTree()
{
    RENDERER_LOG_INFO(
        "deleting triangle tree #" FMT_UNIQUE_ID "...",
        m_triangle_tree_uid);

    // Delete intersection filters.
    for (size_t i = 0; i < m_intersection_filters.size(); ++i)
        delete m_intersection_filters[i];
}

size_t TriangleTree::get_memory_size() const
{
    return
          TreeType::get_memory_size()
        - sizeof(*static_cast<const TreeType*>(this))
        + sizeof(*this)
        + m_triangle_keys.capacity() * sizeof(TriangleKey)
        + m_leaf_data.capacity() * sizeof(uint8);
}

namespace
{
    template <typename Vector>
    void print_vector_stats(const char* label, const Vector& vec)
    {
        const size_t ItemSize = sizeof(typename Vector::value_type);
        const size_t overhead = vec.capacity() - vec.size();

        RENDERER_LOG_DEBUG(
            "%s: size %s (%s)  capacity %s (%s)  overhead %s (%s)",
            label,
            pretty_uint(vec.size()).c_str(),
            pretty_size(vec.size() * ItemSize).c_str(),
            pretty_uint(vec.capacity()).c_str(),
            pretty_size(vec.capacity() * ItemSize).c_str(),
            pretty_uint(overhead).c_str(),
            pretty_size(overhead * ItemSize).c_str());
    }

    #define RENDERER_LOG_VECTOR_STATS(vec) print_vector_stats(#vec, vec)
}

void TriangleTree::build_bvh(
    const Arguments&    arguments,
    const double        time,
    Statistics&         statistics)
{
    RENDERER_LOG_INFO(
        "collecting geometry for triangle tree #" FMT_UNIQUE_ID "...",
        arguments.m_triangle_tree_uid);

    // Collect triangles intersecting the bounding box of this tree.
    vector<TriangleKey> triangle_keys;
    vector<TriangleVertexInfo> triangle_vertex_infos;
    vector<GAABB3> triangle_bboxes;
    collect_triangles(
        arguments,
        time,
        &triangle_keys,
        &triangle_vertex_infos,
        0,
        &triangle_bboxes);
    RENDERER_LOG_VECTOR_STATS(triangle_keys);
    RENDERER_LOG_VECTOR_STATS(triangle_vertex_infos);
    RENDERER_LOG_VECTOR_STATS(triangle_bboxes);

    RENDERER_LOG_INFO(
        "building bvh triangle tree #" FMT_UNIQUE_ID " (%s %s)...",
        arguments.m_triangle_tree_uid,
        pretty_int(triangle_keys.size()).c_str(),
        plural(triangle_keys.size(), "triangle").c_str());

    // Create the partitioner.
    typedef bvh::SAHPartitioner<vector<GAABB3> > Partitioner;
    Partitioner partitioner(
        triangle_bboxes,
        TriangleTreeMaxLeafSize,
        TriangleTreeInteriorNodeTraversalCost,
        TriangleTreeTriangleIntersectionCost);

    // Build the tree.
    typedef bvh::Builder<TriangleTree, Partitioner> Builder;
    Builder builder;
    builder.build<DefaultWallclockTimer>(*this, partitioner, triangle_keys.size(), TriangleTreeMaxLeafSize);
    statistics.insert_time("build time", builder.get_build_time());
    statistics.merge(bvh::TreeStatistics<TriangleTree>(*this, AABB3d(arguments.m_bbox)));

    // Bounding boxes are no longer needed.
    clear_release_memory(triangle_bboxes);

    // Collect triangle vertices.
    vector<GVector3> triangle_vertices;
    collect_triangles<GAABB3>(
        arguments,
        time,
        0,
        0,
        &triangle_vertices,
        0);
    RENDERER_LOG_VECTOR_STATS(triangle_vertices);

    // Compute and propagate motion bounding boxes.
    compute_motion_bboxes(
        partitioner.get_item_ordering(),
        triangle_vertex_infos,
        triangle_vertices,
        0);

    // Store triangles and triangle keys into the tree.
    store_triangles(
        partitioner.get_item_ordering(),
        triangle_vertex_infos,
        triangle_vertices,
        triangle_keys,
        statistics);
}

void TriangleTree::build_sbvh(
    const Arguments&    arguments,
    const double        time,
    Statistics&         statistics)
{
    RENDERER_LOG_INFO(
        "collecting geometry for triangle tree #" FMT_UNIQUE_ID "...",
        arguments.m_triangle_tree_uid);

    // Collect triangles intersecting the bounding box of this tree.
    vector<TriangleKey> triangle_keys;
    vector<TriangleVertexInfo> triangle_vertex_infos;
    vector<GVector3> triangle_vertices;
    vector<AABB3d> triangle_bboxes;
    collect_triangles(
        arguments,
        time,
        &triangle_keys,
        &triangle_vertex_infos,
        &triangle_vertices,
        &triangle_bboxes);
    RENDERER_LOG_VECTOR_STATS(triangle_keys);
    RENDERER_LOG_VECTOR_STATS(triangle_vertex_infos);
    RENDERER_LOG_VECTOR_STATS(triangle_vertices);
    RENDERER_LOG_VECTOR_STATS(triangle_bboxes);

    RENDERER_LOG_INFO(
        "building sbvh triangle tree #" FMT_UNIQUE_ID " (%s %s)...",
        arguments.m_triangle_tree_uid,
        pretty_int(triangle_keys.size()).c_str(),
        plural(triangle_keys.size(), "triangle").c_str());

    // Create the partitioner.
    typedef bvh::SBVHPartitioner<TriangleItemHandler, vector<AABB3d> > Partitioner;
    TriangleItemHandler triangle_handler(
        triangle_vertex_infos,
        triangle_vertices,
        triangle_bboxes);
    Partitioner partitioner(
        triangle_handler,
        triangle_bboxes,
        TriangleTreeMaxLeafSize,
        TriangleTreeBinCount,
        TriangleTreeInteriorNodeTraversalCost,
        TriangleTreeTriangleIntersectionCost);

    // Create the root leaf.
    Partitioner::LeafType* root_leaf = partitioner.create_root_leaf();
    const AABB3d root_leaf_bbox = partitioner.compute_leaf_bbox(*root_leaf);

    // Build the tree.
    typedef bvh::SpatialBuilder<TriangleTree, Partitioner> Builder;
    Builder builder;
    builder.build<DefaultWallclockTimer>(
        *this,
        partitioner,
        root_leaf,
        root_leaf_bbox);
    statistics.insert_time("build time", builder.get_build_time());
    statistics.merge(bvh::TreeStatistics<TriangleTree>(*this, AABB3d(arguments.m_bbox)));

    // Add splits statistics.
    const size_t spatial_splits = partitioner.get_spatial_split_count();
    const size_t object_splits = partitioner.get_object_split_count();
    const size_t total_splits = spatial_splits + object_splits; 
    statistics.insert(
        "splits",
        "spatial " + pretty_uint(spatial_splits) + " (" + pretty_percent(spatial_splits, total_splits) + ")  "
        "object " + pretty_uint(object_splits) + " (" + pretty_percent(object_splits, total_splits) + ")");

    // Bounding boxes are no longer needed.
    clear_release_memory(triangle_bboxes);

    // Compute and propagate motion bounding boxes.
    compute_motion_bboxes(
        partitioner.get_item_ordering(),
        triangle_vertex_infos,
        triangle_vertices,
        0);

    // Store triangles and triangle keys into the tree.
    store_triangles(
        partitioner.get_item_ordering(),
        triangle_vertex_infos,
        triangle_vertices,
        triangle_keys,
        statistics);
}

namespace
{
    //
    // If the bounding box is seen as a flat array of scalars, the swizzle() function converts
    // the bounding box from
    //
    //   min.x  min.y  min.z  max.x  max.y  max.z
    //
    // to
    //
    //   min.x  max.x  min.y  max.y  min.z  max.z
    //

    template <typename T, size_t N>
    AABB<T, N> swizzle(const AABB<T, N>& bbox)
    {
        AABB<T, N> result;
        T* flat_result = &result[0][0];

        for (size_t i = 0; i < N; ++i)
        {
            flat_result[i * 2 + 0] = bbox[0][i];
            flat_result[i * 2 + 1] = bbox[1][i];
        }

        return result;
    }
}

vector<GAABB3> TriangleTree::compute_motion_bboxes(
    const vector<size_t>&               triangle_indices,
    const vector<TriangleVertexInfo>&   triangle_vertex_infos,
    const vector<GVector3>&             triangle_vertices,
    const size_t                        node_index)
{
    NodeType& node = m_nodes[node_index];

    if (node.is_interior())
    {
        const vector<GAABB3> left_bboxes =
            compute_motion_bboxes(
                triangle_indices,
                triangle_vertex_infos,
                triangle_vertices,
                node.get_child_node_index() + 0);

        const vector<GAABB3> right_bboxes =
            compute_motion_bboxes(
                triangle_indices,
                triangle_vertex_infos,
                triangle_vertices,
                node.get_child_node_index() + 1);

        node.set_left_bbox_count(left_bboxes.size());
        node.set_right_bbox_count(right_bboxes.size());

        if (left_bboxes.size() > 1)
        {
            node.set_left_bbox_index(m_node_bboxes.size());

            for (vector<GAABB3>::const_iterator i = left_bboxes.begin(); i != left_bboxes.end(); ++i)
                m_node_bboxes.push_back(swizzle(AABB3d(*i)));
        }

        if (right_bboxes.size() > 1)
        {
            node.set_right_bbox_index(m_node_bboxes.size());

            for (vector<GAABB3>::const_iterator i = right_bboxes.begin(); i != right_bboxes.end(); ++i)
                m_node_bboxes.push_back(swizzle(AABB3d(*i)));
        }

        const size_t bbox_count = max(left_bboxes.size(), right_bboxes.size());
        vector<GAABB3> bboxes(bbox_count);

        for (size_t i = 0; i < bbox_count; ++i)
        {
            bboxes[i] = left_bboxes[i * left_bboxes.size() / bbox_count];
            bboxes[i].insert(right_bboxes[i * right_bboxes.size() / bbox_count]);
        }

        return bboxes;
    }
    else
    {
        const size_t item_begin = node.get_item_index();
        const size_t item_count = node.get_item_count();

        size_t max_motion_segment_count = 0;

        GAABB3 base_pose_bbox;
        base_pose_bbox.invalidate();

        for (size_t i = 0; i < item_count; ++i)
        {
            const size_t triangle_index = triangle_indices[item_begin + i];
            const TriangleVertexInfo& vertex_info = triangle_vertex_infos[triangle_index];

            assert(is_pow2(vertex_info.m_motion_segment_count + 1));

            if (max_motion_segment_count < vertex_info.m_motion_segment_count)
                max_motion_segment_count = vertex_info.m_motion_segment_count;

            base_pose_bbox.insert(triangle_vertices[vertex_info.m_vertex_index + 0]);
            base_pose_bbox.insert(triangle_vertices[vertex_info.m_vertex_index + 1]);
            base_pose_bbox.insert(triangle_vertices[vertex_info.m_vertex_index + 2]);
        }

        vector<GAABB3> bboxes(max_motion_segment_count + 1);
        bboxes[0] = base_pose_bbox;

        if (max_motion_segment_count > 0)
        {
            for (size_t m = 0; m < max_motion_segment_count - 1; ++m)
            {
                bboxes[m + 1].invalidate();

                const double time = static_cast<double>(m + 1) / max_motion_segment_count;

                for (size_t i = 0; i < item_count; ++i)
                {
                    const size_t triangle_index = triangle_indices[item_begin + i];
                    const TriangleVertexInfo& vertex_info = triangle_vertex_infos[triangle_index];

                    const size_t prev_pose_index = truncate<size_t>(time * vertex_info.m_motion_segment_count);
                    const size_t base_vertex_index = vertex_info.m_vertex_index + prev_pose_index * 3;
                    const GScalar k = static_cast<GScalar>(time * vertex_info.m_motion_segment_count - prev_pose_index);

                    bboxes[m + 1].insert(lerp(triangle_vertices[base_vertex_index + 0], triangle_vertices[base_vertex_index + 3], k));
                    bboxes[m + 1].insert(lerp(triangle_vertices[base_vertex_index + 1], triangle_vertices[base_vertex_index + 4], k));
                    bboxes[m + 1].insert(lerp(triangle_vertices[base_vertex_index + 2], triangle_vertices[base_vertex_index + 5], k));
                }
            }

            bboxes[max_motion_segment_count].invalidate();

            for (size_t i = 0; i < item_count; ++i)
            {
                const size_t triangle_index = triangle_indices[item_begin + i];
                const TriangleVertexInfo& vertex_info = triangle_vertex_infos[triangle_index];
                const size_t base_vertex_index = vertex_info.m_vertex_index + vertex_info.m_motion_segment_count * 3;

                bboxes[max_motion_segment_count].insert(triangle_vertices[base_vertex_index + 0]);
                bboxes[max_motion_segment_count].insert(triangle_vertices[base_vertex_index + 1]);
                bboxes[max_motion_segment_count].insert(triangle_vertices[base_vertex_index + 2]);
            }
        }

        return bboxes;
    }
}

void TriangleTree::store_triangles(
    const vector<size_t>&               triangle_indices,
    const vector<TriangleVertexInfo>&   triangle_vertex_infos,
    const vector<GVector3>&             triangle_vertices,
    const vector<TriangleKey>&          triangle_keys,
    Statistics&                         statistics)
{
    const size_t node_count = m_nodes.size();

    // Gather statistics.

    size_t leaf_count = 0;
    size_t fat_leaf_count = 0;
    size_t leaf_data_size = 0;

    for (size_t i = 0; i < node_count; ++i)
    {
        const NodeType& node = m_nodes[i];

        if (node.is_leaf())
        {
            ++leaf_count;

            const size_t item_begin = node.get_item_index();
            const size_t item_count = node.get_item_count();

            const size_t leaf_size =
                TriangleEncoder::compute_size(
                    triangle_vertex_infos,
                    triangle_indices,
                    item_begin,
                    item_count);

            if (leaf_size < NodeType::MaxUserDataSize)
                ++fat_leaf_count;
            else leaf_data_size += leaf_size;
        }
    }

    // Store triangle keys and triangles.

    m_triangle_keys.reserve(triangle_indices.size());
    m_leaf_data.resize(leaf_data_size);

    MemoryWriter leaf_data_writer(m_leaf_data.empty() ? 0 : &m_leaf_data[0]);

    for (size_t i = 0; i < node_count; ++i)
    {
        NodeType& node = m_nodes[i];

        if (node.is_leaf())
        {
            const size_t item_begin = node.get_item_index();
            const size_t item_count = node.get_item_count();

            node.set_item_index(m_triangle_keys.size());

            for (size_t j = 0; j < item_count; ++j)
            {
                const size_t triangle_index = triangle_indices[item_begin + j];
                m_triangle_keys.push_back(triangle_keys[triangle_index]);
            }

            const size_t leaf_size =
                TriangleEncoder::compute_size(
                    triangle_vertex_infos,
                    triangle_indices,
                    item_begin,
                    item_count);

            MemoryWriter user_data_writer(&node.get_user_data<uint8>());

            if (leaf_size <= NodeType::MaxUserDataSize - 4)
            {
                user_data_writer.write<uint32>(~0);

                TriangleEncoder::encode(
                    triangle_vertex_infos,
                    triangle_vertices,
                    triangle_indices,
                    item_begin,
                    item_count,
                    user_data_writer);
            }
            else
            {
                user_data_writer.write(static_cast<uint32>(leaf_data_writer.offset()));

                TriangleEncoder::encode(
                    triangle_vertex_infos,
                    triangle_vertices,
                    triangle_indices,
                    item_begin,
                    item_count,
                    leaf_data_writer);
            }
        }
    }

    statistics.insert_percent("fat leaves", fat_leaf_count, leaf_count);
}

void TriangleTree::create_intersection_filters(
    const Arguments&    arguments,
    Statistics&         statistics)
{
    // Collect object instances.
    vector<size_t> object_instance_indices;
    object_instance_indices.reserve(arguments.m_regions.size());
    for (const_each<RegionInfoVector> i = arguments.m_regions; i; ++i)
        object_instance_indices.push_back(i->get_object_instance_index());

    // Unique the list of object instances.
    sort(object_instance_indices.begin(), object_instance_indices.end());
    object_instance_indices.erase(
        unique(object_instance_indices.begin(), object_instance_indices.end()),
        object_instance_indices.end());

    TextureStore texture_store(arguments.m_scene);
    TextureCache texture_cache(texture_store);

    size_t intersection_filter_count = 0;

    for (const_each<vector<size_t> > i = object_instance_indices; i; ++i)
    {
        // Retrieve the object instance.
        const size_t object_instance_index = *i;
        const ObjectInstance* object_instance =
            arguments.m_assembly.object_instances().get_by_index(object_instance_index);
        assert(object_instance);

        // No intersection filter for this object instance if it doesn't have any front materials.
        if (object_instance->get_front_materials().empty())
            continue;

        // No intersection filter for this object instance if its first front material doesn't have an alpha map.
        const Material* material = object_instance->get_front_materials()[0];
        const Source* alpha_map = material->get_alpha_map();
        if (alpha_map == 0)
            continue;

        // Create an intersection filter for this object instance.
        auto_ptr<IntersectionFilter> intersection_filter(
            new IntersectionFilter(
                arguments.m_scene,
                arguments.m_assembly,
                object_instance_index,
                texture_cache));

        // No intersection filter for this object instance if its alpha map is mostly opaque or semi-transparent.
        if (intersection_filter->get_transparent_pixel_ratio() < 5.0 / 100)
            continue;

        // Allocate the array of intersection filters.
        if (m_intersection_filters.empty())
            m_intersection_filters.resize(object_instance_indices.back() + 1);

        // Store the intersection filter.
        m_intersection_filters[object_instance_index] = intersection_filter.release();
        ++intersection_filter_count;
    }

    statistics.insert<uint64>("inter. filters", intersection_filter_count);
}


//
// TriangleTreeFactory class implementation.
//

TriangleTreeFactory::TriangleTreeFactory(const TriangleTree::Arguments& arguments)
  : m_arguments(arguments)
{
}

auto_ptr<TriangleTree> TriangleTreeFactory::create()
{
    return auto_ptr<TriangleTree>(new TriangleTree(m_arguments));
}

}   // namespace renderer
