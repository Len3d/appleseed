
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

#ifndef APPLESEED_RENDERER_MODELING_SCENE_TEXTUREINSTANCE_H
#define APPLESEED_RENDERER_MODELING_SCENE_TEXTUREINSTANCE_H

// appleseed.renderer headers.
#include "renderer/modeling/entity/entity.h"
#include "renderer/modeling/scene/containers.h"

// appleseed.foundation headers.
#include "foundation/platform/compiler.h"
#include "foundation/utility/autoreleaseptr.h"

// appleseed.main headers.
#include "main/dllsymbol.h"

// Forward declarations.
namespace foundation    { class DictionaryArray; }
namespace foundation    { class LightingConditions; }
namespace renderer      { class ParamArray; }
namespace renderer      { class Texture; }

namespace renderer
{

//
// Texture mapping modes.
//

enum TextureAddressingMode
{
    TextureAddressingClamp = 0,
    TextureAddressingWrap
};

enum TextureFilteringMode
{
    TextureFilteringNearest = 0,
    TextureFilteringBilinear,
    TextureFilteringBicubic,
    TextureFilteringFeline,             // Reference: http://www.hpl.hp.com/techreports/Compaq-DEC/WRL-99-1.pdf
    TextureFilteringEWA
};


//
// An instance of a texture.
//
// todo: allow to specify the lighting conditions of a texture.
//

class DLLSYMBOL TextureInstance
  : public Entity
{
  public:
    // Delete this instance.
    virtual void release() override;

    // Return the name of the instantiated texture in the parent scene or assembly.
    const char* get_texture_name() const;

    // Return the texture mapping modes.
    TextureAddressingMode get_addressing_mode() const;
    TextureFilteringMode get_filtering_mode() const;

    // Return the lighting conditions of the texture.
    const foundation::LightingConditions& get_lighting_conditions() const;

    // Perform entity binding.
    void bind_entities(const TextureContainer& textures);

    // Return the instantiated texture.
    Texture* get_texture() const;

  private:
    friend class TextureInstanceFactory;

    struct Impl;
    Impl* impl;

    TextureAddressingMode           m_addressing_mode;
    TextureFilteringMode            m_filtering_mode;
    Texture*                        m_texture;

    // Constructor.
    TextureInstance(
        const char*                 name,
        const ParamArray&           params,
        const char*                 texture_name);

    // Destructor.
    ~TextureInstance();
};


//
// Texture instance factory.
//

class DLLSYMBOL TextureInstanceFactory
{
  public:
    // Return a set of widget definitions for this texture instance entity model.
    static foundation::DictionaryArray get_widget_definitions();

    // Create a new texture instance.
    static foundation::auto_release_ptr<TextureInstance> create(
        const char*                 name,
        const ParamArray&           params,
        const char*                 texture_name);
};


//
// TextureInstance class implementation.
//

inline TextureAddressingMode TextureInstance::get_addressing_mode() const
{
    return m_addressing_mode;
}

inline TextureFilteringMode TextureInstance::get_filtering_mode() const
{
    return m_filtering_mode;
}

inline Texture* TextureInstance::get_texture() const
{
    return m_texture;
}

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_MODELING_SCENE_TEXTUREINSTANCE_H
