
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
#include "frame.h"

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/kernel/aov/imagestack.h"
#include "renderer/utility/paramarray.h"

// appleseed.foundation headers.
#include "foundation/core/exceptions/exception.h"
#include "foundation/core/exceptions/exceptionioerror.h"
#include "foundation/core/exceptions/exceptionunsupportedfileformat.h"
#include "foundation/image/color.h"
#include "foundation/image/exrimagefilewriter.h"
#include "foundation/image/genericimagefilewriter.h"
#include "foundation/image/image.h"
#include "foundation/image/imageattributes.h"
#include "foundation/image/pixel.h"
#include "foundation/image/tile.h"
#include "foundation/math/scalar.h"
#ifdef APPLESEED_FOUNDATION_USE_SSE
#include "foundation/platform/sse.h"
#endif
#include "foundation/platform/timer.h"
#include "foundation/utility/iostreamop.h"
#include "foundation/utility/otherwise.h"
#include "foundation/utility/stopwatch.h"
#include "foundation/utility/string.h"
#include "foundation/utility/uid.h"

// boost headers.
#include "boost/filesystem/path.hpp"

// Standard headers.
#include <cmath>
#include <memory>
#include <string>

using namespace boost;
using namespace foundation;
using namespace std;

namespace renderer
{

//
// Frame class implementation.
//

struct Frame::Impl
{
    size_t                  m_frame_width;
    size_t                  m_frame_height;
    size_t                  m_tile_width;
    size_t                  m_tile_height;
    PixelFormat             m_pixel_format;
    bool                    m_clamp;
    float                   m_target_gamma;
    float                   m_rcp_target_gamma;
    LightingConditions      m_lighting_conditions;

    auto_ptr<Image>         m_image;
    auto_ptr<ImageStack>    m_aov_images;

    Impl()
      : m_lighting_conditions(IlluminantCIED65, XYZCMFCIE196410Deg)
    {
    }
};

namespace
{
    const UniqueID g_class_uid = new_guid();
}

Frame::Frame(
    const char*         name,
    const ParamArray&   params)
  : Entity(g_class_uid, params)
  , impl(new Impl())
{
    set_name(name);
    extract_parameters();

    // Create the underlying image.
    impl->m_image.reset(
        new Image(
            impl->m_frame_width,
            impl->m_frame_height,
            impl->m_tile_width,
            impl->m_tile_height,
            4,
            impl->m_pixel_format));

    // Retrieve the image properties.
    m_props = impl->m_image->properties();

    // Create the image stack for AOVs.
    impl->m_aov_images.reset(
        new ImageStack(
            impl->m_frame_width,
            impl->m_frame_height,
            impl->m_tile_width,
            impl->m_tile_height));
}

Frame::~Frame()
{
    delete impl;
}

void Frame::release()
{
    delete this;
}

Image& Frame::image() const
{
    return *impl->m_image.get();
}

ImageStack& Frame::aov_images() const
{
    return *impl->m_aov_images.get();
}

const LightingConditions& Frame::get_lighting_conditions() const
{
    return impl->m_lighting_conditions;
}

namespace
{
    template <
        int  ColorSpace,
        bool Clamp,
        bool GammaCorrect
    >
    void transform_generic_tile(Tile& tile, const float rcp_target_gamma)
    {
        assert(tile.get_channel_count() == 4);

        const size_t pixel_count = tile.get_pixel_count();

        for (size_t i = 0; i < pixel_count; ++i)
        {
            // Load the pixel color.
            Color4f color;
            tile.get_pixel(i, color);

            // Apply color space conversion.
            switch (ColorSpace)
            {
              case ColorSpaceSRGB:
                color.rgb() = fast_linear_rgb_to_srgb(color.rgb());
                break;

              case ColorSpaceCIEXYZ:
                color.rgb() = linear_rgb_to_ciexyz(color.rgb());
                break;
            }

            // Apply clamping.
            // todo: mark clamped pixels in the diagnostic map.
            color = Clamp ? saturate(color) : clamp_low(color, 0.0f);

            // Apply gamma correction.
            if (GammaCorrect)
            {
                const float old_alpha = color[3];
                fast_pow_refined(&color[0], rcp_target_gamma);
                color[3] = old_alpha;
            }

            // Store the pixel color.
            tile.set_pixel(i, color);
        }
    }

#ifdef APPLESEED_FOUNDATION_USE_SSE

    template <
        int  ColorSpace,
        bool Clamp,
        bool GammaCorrect
    >
    void transform_float_tile(Tile& tile, const float rcp_target_gamma)
    {
        assert(tile.get_channel_count() == 4);

        float* pixel_ptr = reinterpret_cast<float*>(tile.pixel(0));
        float* pixel_end = pixel_ptr + tile.get_pixel_count() * 4;

        for (; pixel_ptr < pixel_end; pixel_ptr += 4)
        {
            // Load the pixel color.
            sse4f color = loadps(pixel_ptr);
            sse4f original_color = color;

            // Apply color space conversion.
            if (ColorSpace == ColorSpaceSRGB)
            {
                sse4f x = _mm_cvtepi32_ps(_mm_castps_si128(color));

                const sse4f K = set1ps(127.0f);
                x = mulps(x, set1ps(0.1192092896e-6f));     // x *= pow(2.0f, -23)
                x = subps(x, K);

                // One Newton-Raphson refinement step.
                sse4f z = subps(x, floorps(x));
                z = subps(z, mulps(z, z));
                z = mulps(z, set1ps(0.346607f));
                x = addps(x, z);

                x = mulps(x, set1ps(1.0f / 2.4f));

                sse4f y = subps(x, floorps(x));
                y = subps(y, mulps(y, y));
                y = mulps(y, set1ps(0.33971f));
                y = subps(addps(x, K), y);
                y = mulps(y, set1ps(8388608.0f));           // y *= pow(2.0f, 23)

                y = _mm_castsi128_ps(_mm_cvtps_epi32(y));

                // Compute both outcomes of the branch.
                const sse4f a = mulps(set1ps(12.92f), color);
                const sse4f b = subps(mulps(set1ps(1.055f), y), set1ps(0.055f));

                // Interleave them based on the actual comparison.
                const sse4f mask = cmpleps(color, set1ps(0.0031308f));
                color = addps(andps(mask, a), andnotps(mask, b));
            }

            // Apply clamping.
            // todo: mark clamped pixels in the diagnostic map.
            if (Clamp)
                color = minps(maxps(color, set1ps(0.0f)), set1ps(1.0f));
            else color = maxps(color, set1ps(0.0f));

            // Apply gamma correction.
            if (GammaCorrect)
            {
                sse4f x = _mm_cvtepi32_ps(_mm_castps_si128(color));

                const sse4f K = set1ps(127.0f);
                x = mulps(x, set1ps(0.1192092896e-6f));     // x *= pow(2.0f, -23)
                x = subps(x, K);

                // One Newton-Raphson refinement step.
                sse4f z = subps(x, floorps(x));
                z = subps(z, mulps(z, z));
                z = mulps(z, set1ps(0.346607f));
                x = addps(x, z);

                x = mulps(x, set1ps(rcp_target_gamma));

                sse4f y = subps(x, floorps(x));
                y = subps(y, mulps(y, y));
                y = mulps(y, set1ps(0.33971f));
                y = subps(addps(x, K), y);
                y = mulps(y, set1ps(8388608.0f));           // y *= pow(2.0f, 23)

                color = _mm_castsi128_ps(_mm_cvtps_epi32(y));
            }

            // Store the pixel color (leave the alpha channel unmodified).
            storeps(pixel_ptr, shuffleps(color, _mm_unpackhi_ps(color, original_color), _MM_SHUFFLE(3, 0, 1, 0)));
        }
    }

#else

    template <
        int  ColorSpace,
        bool Clamp,
        bool GammaCorrect
    >
    void transform_float_tile(Tile& tile, const float rcp_target_gamma)
    {
        assert(tile.get_channel_count() == 4);

        Color4f* pixel_ptr = reinterpret_cast<Color4f*>(tile.pixel(0));
        Color4f* pixel_end = pixel_ptr + tile.get_pixel_count();

        for (; pixel_ptr < pixel_end; ++pixel_ptr)
        {
            // Load the pixel color.
            Color4f color(*pixel_ptr);

            // Apply color space conversion.
            if (ColorSpace == ColorSpaceSRGB)
                color.rgb() = fast_linear_rgb_to_srgb(color.rgb());

            // Apply clamping.
            // todo: mark clamped pixels in the diagnostic map.
            color = Clamp ? saturate(color) : clamp_to_zero(color);

            // Apply gamma correction.
            if (GammaCorrect)
            {
                const float old_alpha = color[3];
                fast_pow_refined(&color[0], rcp_target_gamma);
                color[3] = old_alpha;
            }

            // Store the pixel color.
            *pixel_ptr = color;
        }
    }

#endif
}

void Frame::transform_to_output_color_space(Tile& tile) const
{
    #define TRANSFORM_GENERIC_TILE(ColorSpace)                                                      \
        if (impl->m_clamp)                                                                          \
        {                                                                                           \
            if (impl->m_target_gamma != 1.0f)                                                       \
                transform_generic_tile<ColorSpace, true, true>(tile, impl->m_rcp_target_gamma);     \
            else transform_generic_tile<ColorSpace, true, false>(tile, impl->m_rcp_target_gamma);   \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            if (impl->m_target_gamma != 1.0f)                                                       \
                transform_generic_tile<ColorSpace, false, true>(tile, impl->m_rcp_target_gamma);    \
            else transform_generic_tile<ColorSpace, false, false>(tile, impl->m_rcp_target_gamma);  \
        }

    #define TRANSFORM_FLOAT_TILE(ColorSpace)                                                        \
        if (impl->m_clamp)                                                                          \
        {                                                                                           \
            if (impl->m_target_gamma != 1.0f)                                                       \
                transform_float_tile<ColorSpace, true, true>(tile, impl->m_rcp_target_gamma);       \
            else transform_float_tile<ColorSpace, true, false>(tile, impl->m_rcp_target_gamma);     \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            if (impl->m_target_gamma != 1.0f)                                                       \
                transform_float_tile<ColorSpace, false, true>(tile, impl->m_rcp_target_gamma);      \
            else transform_float_tile<ColorSpace, false, false>(tile, impl->m_rcp_target_gamma);    \
        }

    if (impl->m_pixel_format == PixelFormatFloat)
    {
        switch (m_color_space)
        {
          case ColorSpaceLinearRGB:
            TRANSFORM_FLOAT_TILE(ColorSpaceLinearRGB);
            break;

          case ColorSpaceSRGB:
            TRANSFORM_FLOAT_TILE(ColorSpaceSRGB);
            break;

          case ColorSpaceCIEXYZ:
            TRANSFORM_GENERIC_TILE(ColorSpaceCIEXYZ);
            break;

          assert_otherwise;
        }
    }
    else
    {
        switch (m_color_space)
        {
          case ColorSpaceLinearRGB:
            TRANSFORM_GENERIC_TILE(ColorSpaceLinearRGB);
            break;

          case ColorSpaceSRGB:
            TRANSFORM_GENERIC_TILE(ColorSpaceSRGB);
            break;

          case ColorSpaceCIEXYZ:
            TRANSFORM_GENERIC_TILE(ColorSpaceCIEXYZ);
            break;

          assert_otherwise;
        }
    }

    #undef TRANSFORM_FLOAT_TILE
    #undef TRANSFORM_GENERIC_TILE
}

void Frame::transform_to_output_color_space(Image& image) const
{
    const CanvasProperties& image_props = image.properties();

    for (size_t ty = 0; ty < image_props.m_tile_count_y; ++ty)
    {
        for (size_t tx = 0; tx < image_props.m_tile_count_x; ++tx)
            transform_to_output_color_space(image.tile(tx, ty));
    }
}

bool Frame::write(const char* file_path) const
{
    assert(file_path);

    const ImageAttributes image_attributes =
        ImageAttributes::create_default_attributes();

    bool result = write_image(file_path, *impl->m_image, image_attributes);

    if (!impl->m_aov_images->empty())
    {
        const filesystem::path boost_file_path(file_path);
        const filesystem::path directory = boost_file_path.parent_path();
        const string base_file_name = boost_file_path.stem().string();
        const string extension = boost_file_path.extension().string();

        for (size_t i = 0; i < impl->m_aov_images->size(); ++i)
        {
            const string aov_file_name =
                base_file_name + "." + impl->m_aov_images->get_name(i) + extension;
            const string aov_file_path = (directory / aov_file_name).string();

            if (!write_image(
                    aov_file_path.c_str(),
                    impl->m_aov_images->get_image(i),
                    image_attributes))
                result = false;
        }
    }

    return result;
}

bool Frame::archive(
    const char*         directory,
    char**              output_path) const
{
    assert(directory);

    // Construct the name of the image file.
    const string filename =
        "autosave." + get_time_stamp_string() + ".exr";

    // Construct the path to the image file.
    const string file_path = (filesystem::path(directory) / filename).string();

    // Return the path to the image file.
    if (output_path)
        *output_path = duplicate_string(file_path.c_str());

    return
        write_image(
            file_path.c_str(),
            *impl->m_image,
            ImageAttributes::create_default_attributes());
}

void Frame::extract_parameters()
{
    // Retrieve frame resolution parameter.
    const Vector2i DefaultResolution(512, 512);
    Vector2i resolution = m_params.get_required<Vector2i>("resolution", DefaultResolution);
    if (resolution[0] < 1 || resolution[1] < 1)
    {
        // Invalid value for resolution parameter, use default.
        RENDERER_LOG_ERROR(
            "invalid value \"%d %d\" for parameter \"%s\", using default value \"%d %d\".",
            resolution[0],
            resolution[1],
            "resolution",
            DefaultResolution[0],
            DefaultResolution[1]);
        resolution = DefaultResolution;
    }
    impl->m_frame_width = static_cast<size_t>(resolution[0]);
    impl->m_frame_height = static_cast<size_t>(resolution[1]);

    // Retrieve tile size parameter.
    const Vector2i DefaultTileSize(32, 32);
    Vector2i tile_size = m_params.get_optional<Vector2i>("tile_size", DefaultTileSize);
    if (tile_size[0] < 1 || tile_size[1] < 1)
    {
        // Invalid value for tile_size parameter, use default.
        RENDERER_LOG_ERROR(
            "invalid value \"%d %d\" for parameter \"%s\", using default value \"%d %d\".",
            tile_size[0],
            tile_size[1],
            "tile_size",
            DefaultTileSize[0],
            DefaultTileSize[1]);
        tile_size = DefaultTileSize;
    }
    impl->m_tile_width = static_cast<size_t>(tile_size[0]);
    impl->m_tile_height = static_cast<size_t>(tile_size[1]);

    // Retrieve pixel format parameter.
    const PixelFormat DefaultPixelFormat = PixelFormatFloat;
    const char* DefaultPixelFormatString = "float";
    const string pixel_format =
        m_params.get_optional<string>("pixel_format", DefaultPixelFormatString);
    if (pixel_format == "uint8")
        impl->m_pixel_format = PixelFormatUInt8;
    else if (pixel_format == "uint16")
        impl->m_pixel_format = PixelFormatUInt16;
    else if (pixel_format == "uint32")
        impl->m_pixel_format = PixelFormatUInt32;
    else if (pixel_format == "half")
        impl->m_pixel_format = PixelFormatHalf;
    else if (pixel_format == "float")
        impl->m_pixel_format = PixelFormatFloat;
    else if (pixel_format == "double")
        impl->m_pixel_format = PixelFormatDouble;
    else
    {
        // Invalid value for pixel_format parameter, use default.
        RENDERER_LOG_ERROR(
            "invalid value \"%s\" for parameter \"%s\", using default value \"%s\".",
            pixel_format.c_str(),
            "pixel_format",
            DefaultPixelFormatString);
        impl->m_pixel_format = DefaultPixelFormat;
    }

    // Retrieve color space parameter.
    const ColorSpace DefaultColorSpace = ColorSpaceLinearRGB;
    const char* DefaultColorSpaceString = "linear_rgb";
    const string color_space =
        m_params.get_optional<string>("color_space", DefaultColorSpaceString);
    if (color_space == "linear_rgb")
        m_color_space = ColorSpaceLinearRGB;
    else if (color_space == "srgb")
        m_color_space = ColorSpaceSRGB;
    else if (color_space == "ciexyz")
        m_color_space = ColorSpaceCIEXYZ;
    else
    {
        // Invalid value for color_space parameter, use default.
        RENDERER_LOG_ERROR(
            "invalid value \"%s\" for parameter \"%s\", using default value \"%s\".",
            color_space.c_str(),
            "color_space",
            DefaultColorSpaceString);
        m_color_space = DefaultColorSpace;
    }

    // Retrieve clamping parameter.
    impl->m_clamp = m_params.get_optional<bool>("clamping", false);

    // Retrieve gamma correction parameter.
    impl->m_target_gamma = m_params.get_optional<float>("gamma_correction", 1.0f);
    impl->m_rcp_target_gamma = 1.0f / impl->m_target_gamma;
}

bool Frame::write_image(
    const char*             file_path,
    const Image&            image,
    const ImageAttributes&  image_attributes) const
{
    assert(file_path);

    Image final_image(image);
    transform_to_output_color_space(final_image);

    Stopwatch<DefaultWallclockTimer> stopwatch;
    stopwatch.start();

    try
    {
        try
        {
            GenericImageFileWriter writer;
            writer.write(file_path, final_image, image_attributes);
        }
        catch (const ExceptionUnsupportedFileFormat&)
        {
            const string extension = lower_case(filesystem::path(file_path).extension().string());

            RENDERER_LOG_ERROR(
                "file format '%s' not supported, writing the image in OpenEXR format "
                "(but keeping the filename unmodified).",
                extension.c_str());

            EXRImageFileWriter writer;
            writer.write(file_path, final_image, image_attributes);
        }
    }
    catch (const ExceptionIOError&)
    {
        RENDERER_LOG_ERROR(
            "failed to write image file %s: i/o error.",
            file_path);

        return false;
    }
    catch (const Exception& e)
    {
        RENDERER_LOG_ERROR(
            "failed to write image file %s: %s.",
            file_path,
            e.what());

        return false;
    }

    stopwatch.measure();

    RENDERER_LOG_INFO(
        "wrote image file %s in %s.",
        file_path,
        pretty_time(stopwatch.get_seconds()).c_str());

    return true;
}


//
// FrameFactory class implementation.
//

auto_release_ptr<Frame> FrameFactory::create(
    const char*         name,
    const ParamArray&   params)
{
    return auto_release_ptr<Frame>(new Frame(name, params));
}

}   // namespace renderer
