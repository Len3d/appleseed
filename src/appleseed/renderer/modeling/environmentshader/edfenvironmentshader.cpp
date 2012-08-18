
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
#include "edfenvironmentshader.h"

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/kernel/shading/shadingresult.h"
#include "renderer/modeling/environmentedf/environmentedf.h"
#include "renderer/modeling/environmentshader/environmentshader.h"
#include "renderer/modeling/project/project.h"
#include "renderer/modeling/scene/containers.h"
#include "renderer/modeling/scene/scene.h"
#include "renderer/utility/paramarray.h"

// appleseed.foundation headers.
#include "foundation/math/vector.h"
#include "foundation/image/colorspace.h"
#include "foundation/utility/containers/dictionary.h"
#include "foundation/utility/containers/specializedarrays.h"

// Standard headers.
#include <string>

// Forward declarations.
namespace renderer  { class InputEvaluator; }

using namespace foundation;
using namespace std;

namespace renderer
{

namespace
{
    //
    // EDF-based environment shader.
    //

    const char* Model = "edf_environment_shader";

    class EDFEnvironmentShader
      : public EnvironmentShader
    {
      public:
        EDFEnvironmentShader(
            const char*             name,
            const ParamArray&       params)
          : EnvironmentShader(name, params)
          , m_env_edf_name(m_params.get_required<string>("environment_edf", ""))
          , m_env_edf(0)
        {
        }

        virtual void release() override
        {
            delete this;
        }

        virtual const char* get_model() const override
        {
            return Model;
        }

        virtual bool on_frame_begin(const Project& project) override
        {
            if (!EnvironmentShader::on_frame_begin(project))
                return false;

            m_env_edf = 0;

            if (!m_env_edf_name.empty())
            {
                m_env_edf =
                    project.get_scene()->environment_edfs().get_by_name(m_env_edf_name.c_str());

                if (m_env_edf == 0)
                {
                    RENDERER_LOG_ERROR(
                        "while preparing environment shader \"%s\": "
                        "cannot find environment EDF \"%s\".",
                        get_name(),
                        m_env_edf_name.c_str());

                    return false;
                }
            }

            return true;
        }

        virtual void evaluate(
            InputEvaluator&         input_evaluator,
            const Vector3d&         direction,
            ShadingResult&          shading_result) const override
        {
            if (m_env_edf)
            {
                // Evaluate the environment EDF.
                shading_result.m_color_space = ColorSpaceSpectral;
                shading_result.m_alpha.set(0.0f);
                m_env_edf->evaluate(
                    input_evaluator,
                    direction,
                    shading_result.m_color);
            }
            else
            {
                // Environment shader not properly initialized: return transparent black.
                shading_result.clear();
            }
        }

      private:
        const string        m_env_edf_name;
        EnvironmentEDF*     m_env_edf;
    };
}


//
// EDFEnvironmentShaderFactory class implementation.
//

const char* EDFEnvironmentShaderFactory::get_model() const
{
    return Model;
}

const char* EDFEnvironmentShaderFactory::get_human_readable_model() const
{
    return "Environment EDF-Based Environment Shader";
}

DictionaryArray EDFEnvironmentShaderFactory::get_widget_definitions() const
{
    DictionaryArray definitions;

    definitions.push_back(
        Dictionary()
            .insert("name", "environment_edf")
            .insert("label", "Environment EDF")
            .insert("widget", "entity_picker")
            .insert("entity_types",
                Dictionary()
                    .insert("environment_edf", "Environment EDFs"))
            .insert("use", "required")
            .insert("default", ""));

    return definitions;
}

auto_release_ptr<EnvironmentShader> EDFEnvironmentShaderFactory::create(
    const char*         name,
    const ParamArray&   params) const
{
    return
        auto_release_ptr<EnvironmentShader>(
            new EDFEnvironmentShader(name, params));
}

}   // namespace renderer
