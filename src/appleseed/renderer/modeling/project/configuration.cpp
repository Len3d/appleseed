
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
#include "configuration.h"

// Standard headers.
#include <cstring>

using namespace foundation;
using namespace std;

namespace renderer
{

//
// Configuration class implementation.
//

namespace
{
    const UniqueID g_class_uid = new_guid();
}

Configuration::Configuration(const char* name)
  : Entity(g_class_uid)
  , m_base(0)
{
    set_name(name);
}

void Configuration::release()
{
    delete this;
}

void Configuration::set_base(const Configuration* base)
{
    m_base = base;
}

const Configuration* Configuration::get_base() const
{
    return m_base;
}

ParamArray Configuration::get_inherited_parameters() const
{
    if (m_base)
    {
        ParamArray params = m_base->m_params;
        params.merge(m_params);
        return params;
    }
    else
    {
        return m_params;
    }
}


//
// ConfigurationFactory class implementation.
//

auto_release_ptr<Configuration> ConfigurationFactory::create(const char* name)
{
    assert(name);

    return auto_release_ptr<Configuration>(new Configuration(name));
}

auto_release_ptr<Configuration> ConfigurationFactory::create(
    const char*         name,
    const ParamArray&   params)
{
    assert(name);

    auto_release_ptr<Configuration> configuration(new Configuration(name));

    configuration->get_parameters().merge(params);

    return configuration;
}


//
// BaseConfigurationFactory class implementation.
//

auto_release_ptr<Configuration> BaseConfigurationFactory::create_base_final()
{
    auto_release_ptr<Configuration> configuration(new Configuration("base_final"));

    ParamArray& parameters = configuration->get_parameters();

    parameters.insert("frame_renderer", "generic");
    parameters.insert("tile_renderer", "generic");
    parameters.insert("sample_renderer", "generic");
    parameters.insert("lighting_engine", "pt");

    ParamArray generic_tile_renderer_params;
    generic_tile_renderer_params.insert("min_samples", "1");
    generic_tile_renderer_params.insert("max_samples", "1");
    generic_tile_renderer_params.insert("sample_filter_size", "4");
    generic_tile_renderer_params.insert("sample_filter_type", "mitchell");
    parameters.dictionaries().insert("generic_tile_renderer", generic_tile_renderer_params);

    return configuration;
}

auto_release_ptr<Configuration> BaseConfigurationFactory::create_base_interactive()
{
    auto_release_ptr<Configuration> configuration(new Configuration("base_interactive"));

    ParamArray& parameters = configuration->get_parameters();

    parameters.insert("frame_renderer", "progressive");
    parameters.insert("sample_renderer", "generic");
    parameters.insert("lighting_engine", "pt");

    // todo: this parameter should be removed as soon as a tile renderer
    // isn't required anymore by renderer::MasterRenderer::render().
    parameters.insert("tile_renderer", "generic");

    // todo: these parameters should be removed as soon as a tile renderer
    // isn't required anymore by renderer::MasterRenderer::render().
    ParamArray generic_tile_renderer_params;
    generic_tile_renderer_params.insert("min_samples", "1");
    generic_tile_renderer_params.insert("max_samples", "1");
    generic_tile_renderer_params.insert("sample_filter_size", "1");
    generic_tile_renderer_params.insert("sample_filter_type", "box");
    parameters.dictionaries().insert("generic_tile_renderer", generic_tile_renderer_params);

    return configuration;
}

bool BaseConfigurationFactory::is_base_configuration(const char* name)
{
    assert(name);

    return strcmp(name, "base_final") == 0 || strcmp(name, "base_interactive") == 0;
}

}   // namespace renderer
