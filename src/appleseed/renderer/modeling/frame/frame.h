
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

#ifndef APPLESEED_RENDERER_MODELING_FRAME_FRAME_H
#define APPLESEED_RENDERER_MODELING_FRAME_FRAME_H

// appleseed.renderer headers.
#include "renderer/modeling/entity/entity.h"

// appleseed.foundation headers.
#include "foundation/image/canvasproperties.h"
#include "foundation/image/colorspace.h"
#include "foundation/math/vector.h"
#include "foundation/platform/compiler.h"
#include "foundation/utility/autoreleaseptr.h"

// appleseed.main headers.
#include "main/dllsymbol.h"

// Standard headers.
#include <cassert>
#include <cstddef>

// Forward declarations.
namespace foundation    { class Image; }
namespace foundation    { class ImageAttributes; }
namespace foundation    { class LightingConditions; }
namespace foundation    { class Tile; }
namespace renderer      { class ImageStack; }
namespace renderer      { class ParamArray; }

namespace renderer
{

//
// Frame class.
//
// Pixels in a frame are always expressed in the linear RGB color space.
//

class DLLSYMBOL Frame
  : public Entity
{
  public:
    // Delete this instance.
    virtual void release() override;

    // Access the main underlying image.
    foundation::Image& image() const;

    // Access the AOV images.
    ImageStack& aov_images() const;

    // Return the normalized device coordinates of a given sample.
    foundation::Vector2d get_sample_position(
        const double    sample_x,               // x coordinate of the sample in the image, in [0,width)
        const double    sample_y) const;        // y coordinate of the sample in the image, in [0,height)
    foundation::Vector2d get_sample_position(
        const size_t    pixel_x,                // x coordinate of the pixel in the image
        const size_t    pixel_y,                // y coordinate of the pixel in the image
        const double    sample_x,               // x coordinate of the sample in the pixel, in [0,1)
        const double    sample_y) const;        // y coordinate of the sample in the pixel, in [0,1)
    foundation::Vector2d get_sample_position(
        const size_t    tile_x,                 // x coordinate of the tile in the image
        const size_t    tile_y,                 // y coordinate of the tile in the image
        const size_t    pixel_x,                // x coordinate of the pixel in the tile
        const size_t    pixel_y,                // y coordinate of the pixel in the tile
        const double    sample_x,               // x coordinate of the sample in the pixel, in [0,1)
        const double    sample_y) const;        // y coordinate of the sample in the pixel, in [0,1)

    // Return the color space the frame should be converted to for display.
    foundation::ColorSpace get_color_space() const;

    // Return the lighting conditions for spectral to RGB conversion.
    const foundation::LightingConditions& get_lighting_conditions() const;

    // Convert a tile or an image from linear RGB to the output color space.
    void transform_to_output_color_space(foundation::Tile& tile) const;
    void transform_to_output_color_space(foundation::Image& image) const;

    // Write the frame to disk.
    // Return true if successful, false otherwise.
    bool write(const char* file_path) const;

    // Archive the frame to a given directory on disk. If output_path is provided,
    // the full path to the output file will be returned. The returned string must
    // be freed using foundation::free_string().
    // Return true if successful, false otherwise.
    bool archive(
        const char*     directory,
        char**          output_path = 0) const;

  private:
    friend class FrameFactory;

    struct Impl;
    Impl* impl;

    foundation::CanvasProperties    m_props;
    foundation::ColorSpace          m_color_space;

    // Constructor.
    Frame(
        const char*         name,
        const ParamArray&   params);

    // Destructor.
    ~Frame();

    void extract_parameters();

    // Write an image to disk after transformation to the frame's color space.
    // Return true if successful, false otherwise.
    bool write_image(
        const char*                         file_path,
        const foundation::Image&            image,
        const foundation::ImageAttributes&  image_attributes) const;
};


//
// FrameFactory class implementation.
//

class DLLSYMBOL FrameFactory
{
  public:
    // Create a new frame.
    static foundation::auto_release_ptr<Frame> create(
        const char*         name,
        const ParamArray&   params);
};


//
// Frame class implementation.
//

inline foundation::ColorSpace Frame::get_color_space() const
{
    return m_color_space;
}

inline foundation::Vector2d Frame::get_sample_position(
    const double    sample_x,
    const double    sample_y) const
{
    return
        foundation::Vector2d(
            sample_x * m_props.m_rcp_canvas_width,
            sample_y * m_props.m_rcp_canvas_height);
}

inline foundation::Vector2d Frame::get_sample_position(
    const size_t    pixel_x,
    const size_t    pixel_y,
    const double    sample_x,
    const double    sample_y) const
{
    return
        get_sample_position(
            pixel_x + sample_x,
            pixel_y + sample_y);
}

inline foundation::Vector2d Frame::get_sample_position(
    const size_t    tile_x,
    const size_t    tile_y,
    const size_t    pixel_x,
    const size_t    pixel_y,
    const double    sample_x,
    const double    sample_y) const
{
    assert(tile_x < m_props.m_tile_count_x);
    assert(tile_y < m_props.m_tile_count_y);

    return
        get_sample_position(
            tile_x * m_props.m_tile_width + pixel_x,
            tile_y * m_props.m_tile_height + pixel_y,
            sample_x,
            sample_y);
}

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_MODELING_FRAME_FRAME_H
