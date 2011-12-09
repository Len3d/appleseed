
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
#include "physicalsurfaceshader.h"

// appleseed.renderer headers.
#include "renderer/kernel/lighting/ilightingengine.h"
#include "renderer/kernel/shading/shadingcontext.h"
#include "renderer/kernel/shading/shadingpoint.h"
#include "renderer/kernel/shading/shadingray.h"
#include "renderer/kernel/shading/shadingresult.h"
#include "renderer/modeling/environment/environment.h"
#include "renderer/modeling/environmentshader/environmentshader.h"
#include "renderer/modeling/input/inputarray.h"
#include "renderer/modeling/input/inputevaluator.h"
#include "renderer/modeling/input/source.h"
#include "renderer/modeling/scene/scene.h"
#include "renderer/modeling/surfaceshader/surfaceshader.h"

// appleseed.foundation headers.
#include "foundation/image/colorspace.h"
#include "foundation/math/vector.h"
#include "foundation/utility/containers/dictionary.h"
#include "foundation/utility/containers/specializedarrays.h"

// Standard headers.
#include <algorithm>
#include <cmath>

// Forward declarations.
namespace renderer  { class Assembly; }
namespace renderer  { class Project; }
namespace renderer  { class TextureCache; }

using namespace foundation;
using namespace std;

namespace renderer
{

namespace
{
    //
    // Physical surface shader.
    //

    const char* Model = "physical_surface_shader";

    class PhysicalSurfaceShader
      : public SurfaceShader
    {
      public:
        PhysicalSurfaceShader(
            const char*             name,
            const ParamArray&       params)
          : SurfaceShader(name, params)
          , m_lighting_conditions(IlluminantCIED65, XYZCMFCIE196410Deg)
          , m_has_alpha_mask(false)
        {
            m_inputs.declare("alpha_mask", InputFormatSpectrum, true);
            m_inputs.declare("aerial_persp_sky_color", InputFormatSpectrum, true);

            const string aerial_persp_mode = m_params.get_optional<string>("aerial_persp_mode", "none");
            if (aerial_persp_mode == "none")
                m_aerial_persp_mode = AerialPerspNone;
            else if (aerial_persp_mode == "environment_shader")
                m_aerial_persp_mode = AerialPerspEnvironmentShader;
            else if (aerial_persp_mode == "sky_color")
                m_aerial_persp_mode = AerialPerspSkyColor;
            else 
            {
                RENDERER_LOG_ERROR(
                    "invalid value \"%s\" for parameter \"aerial_persp_mode\", ",
                    "using default value \"none\"",
                    aerial_persp_mode.c_str());
                m_aerial_persp_mode = AerialPerspNone;
            }

            m_aerial_persp_rcp_distance = 1.0 / m_params.get_optional<double>("aerial_persp_distance", 1000.0);
            m_aerial_persp_intensity = m_params.get_optional<double>("aerial_persp_intensity", 0.01);
        }

        virtual void release() override
        {
            delete this;
        }

        virtual const char* get_model() const override
        {
            return Model;
        }

        virtual void on_frame_begin(
            const Project&          project,
            const Assembly&         assembly) override
        {
            SurfaceShader::on_frame_begin(project, assembly);

            m_has_alpha_mask = m_inputs.source("alpha_mask") != 0;
        }

        virtual void evaluate(
            SamplingContext&        sampling_context,
            const ShadingContext&   shading_context,
            const ShadingPoint&     shading_point,
            ShadingResult&          shading_result) const override
        {
            // Set color space to spectral.
            shading_result.m_color_space = ColorSpaceSpectral;

            // Retrieve the lighting engine.
            ILightingEngine* lighting_engine =
                shading_context.get_lighting_engine();
            assert(lighting_engine);

            // Compute the lighting.
            lighting_engine->compute_lighting(
                sampling_context,
                shading_context,
                shading_point,
                shading_result.m_color);

            // Evaluate alpha mask.
            evaluate_alpha_mask(
                sampling_context,
                shading_context.get_texture_cache(),
                shading_point,
                shading_result.m_alpha);

            if (m_aerial_persp_mode != AerialPerspNone)
            {
                Spectrum sky_color;

                if (m_aerial_persp_mode == AerialPerspSkyColor)
                {
                    // Evaluate the inputs to obtain the sky color.
                    InputValues values;
                    m_inputs.evaluate(
                        shading_context.get_texture_cache(),
                        shading_point.get_input_params(),
                        &values);
                    sky_color = values.m_aerial_persp_sky_color;
                }
                else
                {
                    // Retrieve the environment shader of the scene.
                    const Scene& scene = shading_point.get_scene();
                    const EnvironmentShader* environment_shader =
                        scene.get_environment()->get_environment_shader();

                    if (environment_shader)
                    {
                        // Execute the environment shader to obtain the sky color in the direction of the ray.
                        InputEvaluator input_evaluator(shading_context.get_texture_cache());
                        const ShadingRay& ray = shading_point.get_ray();
                        const Vector3d direction = normalize(ray.m_dir);
                        ShadingResult sky;
                        environment_shader->evaluate(input_evaluator, direction, sky);
                        sky.transform_to_spectrum(m_lighting_conditions);
                        sky_color = sky.m_color;
                    }
                    else sky_color.set(0.0f);
                }

                // Compute the blend factor.
                const double d = shading_point.get_distance() * m_aerial_persp_rcp_distance;
                const double k = m_aerial_persp_intensity * exp(d);
                const double blend = min(k, 1.0);

                // Blend the shading result and the sky color.
                sky_color *= static_cast<float>(blend);
                shading_result.m_color *= static_cast<float>(1.0 - blend);
                shading_result.m_color += sky_color;
            }
        }

        virtual void evaluate_alpha_mask(
            SamplingContext&        sampling_context,
            TextureCache&           texture_cache,
            const ShadingPoint&     shading_point,
            Alpha&                  alpha) const override
        {
            if (!m_has_alpha_mask)
            {
                // Set alpha channel to full opacity.
                alpha = Alpha(1.0);
            }
            else
            {
                // Evaluate the inputs.
                InputValues values;
                m_inputs.evaluate(
                    texture_cache,
                    shading_point.get_input_params(),
                    &values);

                // Set alpha channel.
                alpha = values.m_alpha_mask_alpha;
            }
        }

      private:
        struct InputValues
        {
            Spectrum    m_alpha_mask_color;             // unused
            Alpha       m_alpha_mask_alpha;

            Spectrum    m_aerial_persp_sky_color;
            Alpha       m_aerial_persp_sky_alpha;       // unused
        };

        enum AerialPerspMode
        {
            AerialPerspNone,
            AerialPerspEnvironmentShader,
            AerialPerspSkyColor
        };

        const LightingConditions    m_lighting_conditions;

        bool                        m_has_alpha_mask;

        AerialPerspMode             m_aerial_persp_mode;
        double                      m_aerial_persp_rcp_distance;
        double                      m_aerial_persp_intensity;
    };
}


//
// PhysicalSurfaceShaderFactory class implementation.
//

const char* PhysicalSurfaceShaderFactory::get_model() const
{
    return Model;
}

const char* PhysicalSurfaceShaderFactory::get_human_readable_model() const
{
    return "Physical";
}

DictionaryArray PhysicalSurfaceShaderFactory::get_widget_definitions() const
{
    DictionaryArray definitions;

    definitions.push_back(
        Dictionary()
            .insert("name", "alpha_mask")
            .insert("label", "Alpha Mask")
            .insert("widget", "entity_picker")
            .insert("entity_types",
                Dictionary()
                    .insert("color", "Colors")
                    .insert("texture_instance", "Textures"))
            .insert("use", "optional")
            .insert("default", ""));

    definitions.push_back(
        Dictionary()
            .insert("name", "aerial_persp_mode")
            .insert("label", "Aerial Perspective Mode")
            .insert("widget", "dropdown_list")
            .insert("dropdown_items",
                Dictionary()
                    .insert("None", "none")
                    .insert("Use Environment Shader", "environment_shader")
                    .insert("Use Sky Color", "sky_color"))
            .insert("use", "optional")
            .insert("default", "none"));

    definitions.push_back(
        Dictionary()
            .insert("name", "aerial_persp_sky_color")
            .insert("label", "Aerial Perspective Sky Color")
            .insert("widget", "entity_picker")
            .insert("entity_types", Dictionary().insert("color", "Colors"))
            .insert("use", "optional")
            .insert("default", ""));

    definitions.push_back(
        Dictionary()
            .insert("name", "aerial_persp_distance")
            .insert("label", "Aerial Perspective Distance")
            .insert("widget", "text_box")
            .insert("default", "1000.0")
            .insert("use", "optional"));

    definitions.push_back(
        Dictionary()
            .insert("name", "aerial_persp_intensity")
            .insert("label", "Aerial Perspective Intensity")
            .insert("widget", "text_box")
            .insert("default", "0.01")
            .insert("use", "optional"));

    return definitions;
}

auto_release_ptr<SurfaceShader> PhysicalSurfaceShaderFactory::create(
    const char*         name,
    const ParamArray&   params) const
{
    return
        auto_release_ptr<SurfaceShader>(
            new PhysicalSurfaceShader(name, params));
}

}   // namespace renderer
