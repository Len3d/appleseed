
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
#include "lightsampler.h"

// appleseed.renderer headers.
#include "renderer/kernel/tessellation/statictessellation.h"
#include "renderer/modeling/light/light.h"
#include "renderer/modeling/material/material.h"
#include "renderer/modeling/object/iregion.h"
#include "renderer/modeling/object/object.h"
#include "renderer/modeling/object/regionkit.h"
#include "renderer/modeling/scene/assembly.h"
#include "renderer/modeling/scene/assemblyinstance.h"
#include "renderer/modeling/scene/containers.h"
#include "renderer/modeling/scene/objectinstance.h"
#include "renderer/modeling/scene/scene.h"

// appleseed.foundation headers.
#include "foundation/core/exceptions/exceptionnotimplemented.h"
#include "foundation/math/area.h"
#include "foundation/math/sampling.h"
#include "foundation/utility/foreach.h"
#include "foundation/utility/lazy.h"
#include "foundation/utility/string.h"

// Standard headers.
#include <algorithm>
#include <map>

using namespace foundation;
using namespace std;

namespace renderer
{

//
// LightSampler class implementation.
//

namespace
{
    // Return true if a given assembly references at least one light-emitting material.
    bool has_emitting_materials(const Assembly& assembly)
    {
        for (const_each<MaterialContainer> i = assembly.materials(); i; ++i)
        {
            if (i->get_edf())
                return true;
        }

        return false;
    }

    // Return true if at least one material emits light.
    bool has_emitting_materials(const MaterialArray& materials)
    {
        for (size_t i = 0; i < materials.size(); ++i)
        {
            if (materials[i] && materials[i]->get_edf())
                return true;
        }

        return false;
    }
}

LightSampler::LightSampler(const Scene& scene)
  : m_light_count(0)
  , m_total_emissive_area(0.0)
  , m_rcp_total_emissive_area(0.0)
{
    RENDERER_LOG_INFO("collecting light emitters...");

    // Collect all lights and light emitting triangles.
    collect_lights(scene);
    collect_emitting_triangles(scene);

    // Precompute some values.
    m_light_count = m_lights.size();
    m_rcp_total_emissive_area = 1.0 / m_total_emissive_area;

    // Prepare the CDF for sampling.
    if (m_light_cdf.valid())
        m_light_cdf.prepare();

    RENDERER_LOG_INFO(
        "found %s %s, %s emitting %s",
        pretty_int(m_light_count).c_str(),
        plural(m_light_count, "light").c_str(),
        pretty_int(m_emitting_triangles.size()).c_str(),
        plural(m_emitting_triangles.size(), "triangle").c_str());
}

void LightSampler::collect_lights(const Scene& scene)
{
    for (const_each<AssemblyInstanceContainer> i = scene.assembly_instances(); i; ++i)
        collect_lights(*i);
}

void LightSampler::collect_lights(const AssemblyInstance& assembly_instance)
{
    const Assembly& assembly = assembly_instance.get_assembly();

    for (const_each<LightContainer> i = assembly.lights(); i; ++i)
    {
        // Retrieve the light.
        const Light& light = *i;

        // Copy the light into the light vector.
        const size_t light_index = m_lights.size();
        m_lights.push_back(&light);

        // todo: compute importance.
        const double importance = 1.0;

        // Insert the light into the CDF.
        m_light_cdf.insert(light_index, importance);
    }
}

void LightSampler::collect_emitting_triangles(const Scene& scene)
{
    for (const_each<AssemblyInstanceContainer> i = scene.assembly_instances(); i; ++i)
    {
        const AssemblyInstance& assembly_instance = *i;
        const Assembly& assembly = assembly_instance.get_assembly();

        if (has_emitting_materials(assembly))
            collect_emitting_triangles(assembly_instance, assembly);
    }
}

void LightSampler::collect_emitting_triangles(
    const AssemblyInstance& assembly_instance,
    const Assembly&         assembly)
{
    // Loop over the object instances of the assembly.
    const size_t object_instance_count = assembly.object_instances().size();
    for (size_t object_instance_index = 0; object_instance_index < object_instance_count; ++object_instance_index)
    {
        // Retrieve the object instance.
        const ObjectInstance* object_instance = assembly.object_instances().get_by_index(object_instance_index);

        // Retrieve the materials of the object instance.
        const MaterialArray& materials = object_instance->get_materials();

        // Skip object instances without light emitting materials.
        if (!has_emitting_materials(materials))
            continue;

        // Compute the object space to world space transformation.
        const Transformd& object_instance_transform = object_instance->get_transform();
        const Transformd& assembly_instance_transform = assembly_instance.get_transform();
        const Transformd global_transform = assembly_instance_transform * object_instance_transform;

        // Retrieve the object.
        Object* object = assembly.objects().get_by_index(object_instance->get_object_index());
        assert(object);

        // Retrieve the region kit of the object.
        Access<RegionKit> region_kit(&object->get_region_kit());

        // Loop over the regions of the object.
        const size_t region_count = region_kit->size();
        for (size_t region_index = 0; region_index < region_count; ++region_index)
        {
            // Retrieve the region.
            const IRegion* region = (*region_kit)[region_index];

            // Retrieve the tessellation of the region.
            Access<StaticTriangleTess> tess(&region->get_static_triangle_tess());

            // Loop over the triangles of the region.
            const size_t triangle_count = tess->m_primitives.size();
            for (size_t triangle_index = 0; triangle_index < triangle_count; ++triangle_index)
            {
                // Fetch the triangle.
                const Triangle& triangle = tess->m_primitives[triangle_index];
                const size_t pa_index = static_cast<size_t>(triangle.m_pa);

                // Skip triangles without materials.
                if (pa_index >= materials.size())
                    continue;

                // Fetch the material assigned to this triangle.
                const Material* material = materials[pa_index];
                assert(material);

                // Skip triangles that don't emit light.
                if (material->get_edf() == 0)
                    continue;

                // Retrieve object instance space vertices of the triangle.
                const GVector3& v0_os = tess->m_vertices[triangle.m_v0];
                const GVector3& v1_os = tess->m_vertices[triangle.m_v1];
                const GVector3& v2_os = tess->m_vertices[triangle.m_v2];

                // Transform triangle vertices to assembly space.
                const GVector3 v0_as = object_instance_transform.transform_point_to_parent(v0_os);
                const GVector3 v1_as = object_instance_transform.transform_point_to_parent(v1_os);
                const GVector3 v2_as = object_instance_transform.transform_point_to_parent(v2_os);

                // Compute the support plane of the hit triangle in assembly space.
                const GTriangleType triangle_geometry(v0_as, v1_as, v2_as);
                TriangleSupportPlaneType triangle_support_plane;
                triangle_support_plane.initialize(TriangleType(triangle_geometry));

                // Transform triangle vertices to world space.
                const Vector3d v0(assembly_instance_transform.transform_point_to_parent(v0_as));
                const Vector3d v1(assembly_instance_transform.transform_point_to_parent(v1_as));
                const Vector3d v2(assembly_instance_transform.transform_point_to_parent(v2_as));

                // Compute the geometric normal to the triangle and the area of the triangle.
                Vector3d geometric_normal = cross(v1 - v0, v2 - v0);
                const double geometric_normal_norm = norm(geometric_normal);
                if (geometric_normal_norm == 0.0)
                    continue;
                const double rcp_geometric_normal_norm = 1.0 / geometric_normal_norm;
                const double rcp_area = 2.0 * rcp_geometric_normal_norm;
                const double area = 0.5 * geometric_normal_norm;
                geometric_normal *= rcp_geometric_normal_norm;
                assert(is_normalized(geometric_normal));

                // Keep track of the total area of the light emitting triangles.
                m_total_emissive_area += area;

                // Retrieve object instance space vertex normals.
                const GVector3& n0_os = tess->m_vertex_normals[triangle.m_n0];
                const GVector3& n1_os = tess->m_vertex_normals[triangle.m_n1];
                const GVector3& n2_os = tess->m_vertex_normals[triangle.m_n2];

                // Transform vertex normals to world space.
                const Vector3d n0(normalize(global_transform.transform_normal_to_parent(n0_os)));
                const Vector3d n1(normalize(global_transform.transform_normal_to_parent(n1_os)));
                const Vector3d n2(normalize(global_transform.transform_normal_to_parent(n2_os)));

                const size_t emitting_triangle_index = m_lights.size() + m_emitting_triangles.size();

                // Copy the triangle into the array of emitting triangles.
                EmittingTriangle emitting_triangle;
                emitting_triangle.m_assembly_instance_uid = assembly_instance.get_uid();
                emitting_triangle.m_object_instance_index = object_instance_index;
                emitting_triangle.m_region_index = region_index;
                emitting_triangle.m_triangle_index = triangle_index;
                emitting_triangle.m_v0 = v0;
                emitting_triangle.m_v1 = v1;
                emitting_triangle.m_v2 = v2;
                emitting_triangle.m_n0 = n0;
                emitting_triangle.m_n1 = n1;
                emitting_triangle.m_n2 = n2;
                emitting_triangle.m_geometric_normal = geometric_normal;
                emitting_triangle.m_triangle_support_plane = triangle_support_plane;
                emitting_triangle.m_rcp_area = rcp_area;
                emitting_triangle.m_edf = material->get_edf();
                m_emitting_triangles.push_back(emitting_triangle);

                // Insert the triangle into the CDF.
                m_light_cdf.insert(emitting_triangle_index, area);
            }
        }
    }
}

bool LightSampler::sample(
    const Vector3d&         s,
    LightSample&            sample) const
{
    // No light source in the scene.
    if (!m_light_cdf.valid())
        return false;

    sample_emitters(s, sample);

    return true;
}

bool LightSampler::sample(
    SamplingContext&        sampling_context,
    LightSample&            sample) const
{
    // No light source in the scene.
    if (!m_light_cdf.valid())
        return false;

    sampling_context.split_in_place(3, 1);

    sample_emitters(sampling_context.next_vector2<3>(), sample);

    return true;
}

bool LightSampler::sample(
    SamplingContext&        sampling_context,
    const size_t            sample_count,
    LightSampleVector&      samples) const
{
    // No light source in the scene.
    if (!m_light_cdf.valid())
        return false;

    sampling_context.split_in_place(3, sample_count);

    for (size_t i = 0; i < sample_count; ++i)
    {
        LightSample sample;
        sample_emitters(sampling_context.next_vector2<3>(), sample);
        samples.push_back(sample);
    }

    return true;
}

void LightSampler::sample_emitters(
    const Vector3d&         s,
    LightSample&            sample) const
{
    // Sample the set of emitters (lights and emitting triangles).
    const LightCDF::ItemWeightPair result = m_light_cdf.sample(s[0]);
    const size_t emitter_index = result.first;
    const double emitter_prob = result.second;

    // Generate one sample on the chosen emitter.
    if (emitter_index < m_light_count)
    {
        sample_light(
            Vector2d(s[1], s[2]),
            emitter_index,
            emitter_prob,
            sample);
    }
    else
    {
        sample_emitting_triangle(
            Vector2d(s[1], s[2]),
            emitter_index - m_light_count,
            emitter_prob,
            sample);
    }
}

void LightSampler::sample_light(
    const Vector2d&         s,
    const size_t            light_index,
    const double            light_prob,
    LightSample&            sample) const
{
    // Fetch the light.
    const Light* light = m_lights[light_index];
    assert(light);

    throw ExceptionNotImplemented();
}

void LightSampler::sample_emitting_triangle(
    const Vector2d&         s,
    const size_t            triangle_index,
    const double            triangle_prob,
    LightSample&            sample) const
{
    // Set the triangle index.
    sample.m_triangle_index = triangle_index;

    // Uniformly sample the surface of the triangle.
    const Vector3d bary = sample_triangle_uniform(s);

    // Set the barycentric coordinates.
    sample.m_input_params.m_uv[0] = bary[0];
    sample.m_input_params.m_uv[1] = bary[1];

    // Fetch the emitting triangle.
    const EmittingTriangle& triangle = m_emitting_triangles[triangle_index];

    // Compute the world space position of the sample.
    sample.m_input_params.m_point =
          bary[0] * triangle.m_v0
        + bary[1] * triangle.m_v1
        + bary[2] * triangle.m_v2;

    // Compute the world space shading normal at the position of the sample.
    sample.m_input_params.m_shading_normal =
          bary[0] * triangle.m_n0
        + bary[1] * triangle.m_n1
        + bary[2] * triangle.m_n2;
    sample.m_input_params.m_shading_normal =
        normalize(sample.m_input_params.m_shading_normal);

    // Set the world space geometric normal.
    sample.m_input_params.m_geometric_normal = triangle.m_geometric_normal;

    // Set the EDF.
    sample.m_edf = triangle.m_edf;

    // Compute the probability of choosing this sample.
    assert(feq(triangle_prob * triangle.m_rcp_area, m_rcp_total_emissive_area));
    sample.m_probability = m_rcp_total_emissive_area;
}

}   // namespace renderer
