
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
#include "frame.h"

// appleseed.foundation headers.
#include "foundation/core/exceptions/exception.h"
#include "foundation/core/exceptions/exceptionioerror.h"
#include "foundation/image/colorspace.h"
#include "foundation/image/exrimagefilewriter.h"
#include "foundation/image/genericimagefilewriter.h"
#include "foundation/image/image.h"
#include "foundation/image/imageattributes.h"
#include "foundation/image/pixel.h"
#include "foundation/image/tile.h"
#include "foundation/math/scalar.h"
#include "foundation/utility/stopwatch.h"
#include "foundation/utility/string.h"
#include "foundation/utility/test.h"

// boost headers.
#include "boost/filesystem/path.hpp"

// Standard headers.
#include <algorithm>
#include <cstring>

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
    size_t              m_frame_width;
    size_t              m_frame_height;
    size_t              m_tile_width;
    size_t              m_tile_height;
    PixelFormat         m_pixel_format;
    ColorSpace          m_color_space;
    bool                m_gamma_correct;
    float               m_target_gamma;
    float               m_rcp_target_gamma;

    auto_ptr<Image>     m_image;
    LightingConditions  m_lighting_conditions;

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
}

Frame::~Frame()
{
    delete impl;
}

void Frame::release()
{
    delete this;
}

const CanvasProperties& Frame::properties() const
{
    return m_props;
}

const LightingConditions& Frame::get_lighting_conditions() const
{
    return impl->m_lighting_conditions;
}

Tile& Frame::tile(
    const size_t        tile_x,
    const size_t        tile_y) const
{
    return impl->m_image->tile(tile_x, tile_y);
}

void Frame::transform_tile_to_frame_color_space(Tile& tile) const
{
    assert(tile.get_channel_count() == 4);

    const size_t tile_width = tile.get_width();
    const size_t tile_height = tile.get_height();

    for (size_t y = 0; y < tile_height; ++y)
    {
        for (size_t x = 0; x < tile_width; ++x)
        {
            Color4f linear_rgb;
            tile.get_pixel(x, y, linear_rgb);
            tile.set_pixel(x, y, linear_rgb_to_frame(linear_rgb));
        }
    }
}

void Frame::transform_image_to_frame_color_space(Image& image) const
{
    const CanvasProperties& image_props = image.properties();

    assert(image_props.m_channel_count == 4);

    for (size_t ty = 0; ty < image_props.m_tile_count_y; ++ty)
    {
        for (size_t tx = 0; tx < image_props.m_tile_count_x; ++tx)
        {
            transform_tile_to_frame_color_space(image.tile(tx, ty));
        }
    }
}

namespace
{
    double write_image(
        const char*             filename,
        const Image&            image,
        const ImageAttributes&  image_attributes)
    {
        Stopwatch<DefaultWallclockTimer> stopwatch;
        stopwatch.start();

        try
        {
            GenericImageFileWriter writer;
            writer.write(filename, image, image_attributes);
        }
        catch (const GenericImageFileWriter::ExceptionUnknownFileTypeError&)
        {
            // Extract the extension of the image filename.
            const filesystem::path filepath(filename);
            const string extension = lower_case(filepath.extension());

            // Emit an error message.
            RENDERER_LOG_ERROR(
                "file format '%s' not supported, writing the image in OpenEXR format "
                "(but keeping the filename unmodified)",
                extension.c_str());

            // Write the image in OpenEXR format.
            EXRImageFileWriter writer;
            writer.write(filename, image, image_attributes);
        }

        stopwatch.measure();

        return stopwatch.get_seconds();
    }
}

bool Frame::write(const char* filename) const
{
    assert(filename);

    try
    {
        Image final_image(*impl->m_image);
        transform_image_to_frame_color_space(final_image);

        const double seconds =
            write_image(
                filename,
                final_image,
                ImageAttributes::create_default_attributes());

        RENDERER_LOG_INFO(
            "wrote image file %s in %s", filename, pretty_time(seconds).c_str());
    }
    catch (const ExceptionIOError&)
    {
        RENDERER_LOG_ERROR(
            "failed to write image file %s: i/o error",
            filename);

        return false;
    }
    catch (const Exception& e)
    {
        RENDERER_LOG_ERROR(
            "failed to write image file %s: %s",
            filename,
            e.what());

        return false;
    }

    return true;
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
    const filesystem::path image_path = filesystem::path(directory) / filename;

    // Return the path to the image file.
    if (output_path)
        *output_path = duplicate_string(image_path.file_string().c_str());

    try
    {
        Image final_image(*impl->m_image);
        transform_image_to_frame_color_space(final_image);

        const double seconds =
            write_image(
                image_path.file_string().c_str(),
                final_image,
                ImageAttributes::create_default_attributes());

        RENDERER_LOG_INFO(
            "frame successfully archived to %s in %s",
            image_path.file_string().c_str(),
            pretty_time(seconds).c_str());
    }
    catch (const ExceptionIOError&)
    {
        RENDERER_LOG_WARNING(
            "automatic frame archiving to %s failed: i/o error",
            image_path.file_string().c_str());

        return false;
    }
    catch (const Exception& e)
    {
        RENDERER_LOG_WARNING(
            "automatic frame archiving to %s failed: %s",
            image_path.file_string().c_str(),
            e.what());

        return false;
    }

    return true;
}

namespace
{
    template <typename T, size_t N>
    bool has_nan(const Color<T, N>& color)
    {
        for (size_t i = 0; i < N; ++i)
        {
            if (color[i] != color[i])
                return true;
        }

        return false;
    }
    
    double accumulate_luminance(const Tile& tile)
    {
        double accumulated_luminance = 0.0;

        const size_t tile_width = tile.get_width();
        const size_t tile_height = tile.get_height();

        for (size_t y = 0; y < tile_height; ++y)
        {
            for (size_t x = 0; x < tile_width; ++x)
            {
                // Fetch the pixel color; assume linear RGBA.
                Color4f linear_rgba;
                tile.get_pixel(x, y, linear_rgba);

                // Extract the RGB part (ignore the alpha channel).
                const Color3f linear_rgb = linear_rgba.rgb();

                // Skip pixels containing NaN values.
                if (has_nan(linear_rgb))
                    continue;

                // Compute the Rec. 709 relative luminance of this pixel.
                const float lum = luminance(max(linear_rgb, Color3f(0.0)));

                // It should no longer be possible to have NaN at this point.
                assert(lum == lum);

                accumulated_luminance += static_cast<double>(lum);
            }
        }

        return accumulated_luminance;
    }

    double accumulate_luminance(const Image& image)
    {
        double accumulated_luminance = 0.0;

        const CanvasProperties& props = image.properties();

        for (size_t ty = 0; ty < props.m_tile_count_y; ++ty)
        {
            for (size_t tx = 0; tx < props.m_tile_count_x; ++tx)
            {
                const Tile& tile = image.tile(tx, ty);
                accumulated_luminance += accumulate_luminance(tile);
            }
        }

        return accumulated_luminance;
    }

    double do_compute_average_luminance(const Image& image)
    {
        const double accumulated_luminance = accumulate_luminance(image);

        const CanvasProperties& props = image.properties();
        const double average_luminance = accumulated_luminance / props.m_pixel_count;

        return average_luminance;
    }

    TEST_SUITE(Renderer_Modeling_Frame_Details)
    {
        TEST_CASE(AccumulateLuminance_Given2x2TileFilledWithZeroes_ReturnsZero)
        {
            Tile tile(2, 2, 4, PixelFormatFloat);
            tile.clear(Color4f(0.0f));

            const double accumulated_luminance = accumulate_luminance(tile);

            EXPECT_EQ(0.0, accumulated_luminance);
        }

        TEST_CASE(AccumulateLuminance_Given2x2TileFilledWithOnes_ReturnsFour)
        {
            Tile tile(2, 2, 4, PixelFormatFloat);
            tile.clear(Color4f(1.0f));

            const double accumulated_luminance = accumulate_luminance(tile);

            EXPECT_FEQ(4.0, accumulated_luminance);
        }

        TEST_CASE(AccumulateLuminance_Given2x2TileFilledWithMinusOnes_ReturnsZero)
        {
            Tile tile(2, 2, 4, PixelFormatFloat);
            tile.clear(Color4f(-1.0f));

            const double accumulated_luminance = accumulate_luminance(tile);

            EXPECT_EQ(0.0, accumulated_luminance);
        }

        void clear_image(Image& image, const Color4f& color)
        {
            const CanvasProperties& props = image.properties();

            for (size_t ty = 0; ty < props.m_tile_count_y; ++ty)
            {
                for (size_t tx = 0; tx < props.m_tile_count_x; ++tx)
                {
                    Tile& tile = image.tile(tx, ty);
                    tile.clear(color);
                }
            }
        }

        TEST_CASE(ClearImage_Given4x4Image_FillsImageWithGivenValue)
        {
            const Color4f Expected(42.0f);

            Image image(4, 4, 2, 2, 4, PixelFormatFloat);
            clear_image(image, Expected);

            const CanvasProperties& props = image.properties();

            for (size_t ty = 0; ty < props.m_tile_count_y; ++ty)
            {
                for (size_t tx = 0; tx < props.m_tile_count_x; ++tx)
                {
                    const Tile& tile = image.tile(tx, ty);
                    const size_t tile_width = tile.get_width();
                    const size_t tile_height = tile.get_height();

                    for (size_t y = 0; y < tile_height; ++y)
                    {
                        for (size_t x = 0; x < tile_width; ++x)
                        {
                            Color4f value;
                            tile.get_pixel(x, y, value);

                            EXPECT_EQ(Expected, value);
                        }
                    }
                }
            }
        }

        TEST_CASE(AccumulateLuminance_Given4x4ImageFilledWithZeroes_ReturnsZero)
        {
            Image image(4, 4, 2, 2, 4, PixelFormatFloat);
            clear_image(image, Color4f(0.0f));

            const double accumulated_luminance = accumulate_luminance(image);

            EXPECT_EQ(0.0, accumulated_luminance);
        }

        TEST_CASE(AccumulateLuminance_Given4x4ImageFilledWithOnes_ReturnsSixteen)
        {
            Image image(4, 4, 2, 2, 4, PixelFormatFloat);
            clear_image(image, Color4f(1.0f));

            const double accumulated_luminance = accumulate_luminance(image);

            EXPECT_FEQ(16.0, accumulated_luminance);
        }

        TEST_CASE(DoComputeAverageLuminance_Given4x4ImageFilledWithZeroes_ReturnsZero)
        {
            Image image(4, 4, 2, 2, 4, PixelFormatFloat);
            clear_image(image, Color4f(0.0f));

            const double average_luminance = do_compute_average_luminance(image);

            EXPECT_EQ(0.0, average_luminance);
        }

        TEST_CASE(DoComputeAverageLuminance_Given4x4ImageFilledWithOnes_ReturnsOne)
        {
            Image image(4, 4, 2, 2, 4, PixelFormatFloat);
            clear_image(image, Color4f(1.0f));

            const double average_luminance = do_compute_average_luminance(image);

            EXPECT_FEQ(1.0, average_luminance);
        }
    }
}

double Frame::compute_average_luminance() const
{
    return do_compute_average_luminance(*impl->m_image.get());
}

namespace
{
    struct ExceptionIncompatibleImages
      : public Exception
    {
    };

    double sum_pixel_components(
        const Tile&     tile,
        const size_t    i)
    {
        double sum = 0.0;

        const size_t channel_count = tile.get_channel_count();
        for (size_t c = 0; c < channel_count; ++c)
            sum += tile.get_component<double>(i, c);

        return sum;
    }

    double do_compute_rms_deviation(
        const Image&    image,
        const Image&    ref_image)
    {
        const CanvasProperties& props = image.properties();
        const CanvasProperties& ref_props = ref_image.properties();

        if (props.m_canvas_width != ref_props.m_canvas_width ||
            props.m_canvas_height != ref_props.m_canvas_height ||
            props.m_tile_width != ref_props.m_tile_width ||
            props.m_tile_height != ref_props.m_tile_height ||
            props.m_channel_count != ref_props.m_channel_count)
            throw ExceptionIncompatibleImages();

        double mse = 0.0;   // mean square error

        for (size_t ty = 0; ty < props.m_tile_count_y; ++ty)
        {
            for (size_t tx = 0; tx < props.m_tile_count_x; ++tx)
            {
                const Tile& tile = image.tile(tx, ty);
                const size_t tile_width = tile.get_width();
                const size_t tile_height = tile.get_height();

                const Tile& ref_tile = ref_image.tile(tx, ty);
                assert(ref_tile.get_width() == tile_width);
                assert(ref_tile.get_height() == tile_height);

                for (size_t i = 0; i < tile_width * tile_height; ++i)
                {
                    const double sum = sum_pixel_components(tile, i);
                    const double ref_sum = sum_pixel_components(ref_tile, i);
                    mse += square(sum - ref_sum);
                }
            }
        }

        mse /= props.m_pixel_count * square(props.m_channel_count);

        return sqrt(mse);
    }
}

double Frame::compute_rms_deviation(const Image& ref_image) const
{
    return do_compute_rms_deviation(*impl->m_image.get(), ref_image);
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
            "invalid value \"%d %d\" for parameter \"%s\", using default value \"%d %d\"",
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
            "invalid value \"%d %d\" for parameter \"%s\", using default value \"%d %d\"",
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
            "invalid value \"%s\" for parameter \"%s\", using default value \"%s\"",
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
        impl->m_color_space = ColorSpaceLinearRGB;
    else if (color_space == "srgb")
        impl->m_color_space = ColorSpaceSRGB;
    else if (color_space == "ciexyz")
        impl->m_color_space = ColorSpaceCIEXYZ;
    else
    {
        // Invalid value for color_space parameter, use default.
        RENDERER_LOG_ERROR(
            "invalid value \"%s\" for parameter \"%s\", using default value \"%s\"",
            color_space.c_str(),
            "color_space",
            DefaultColorSpaceString);
        impl->m_color_space = DefaultColorSpace;
    }

    // Retrieve gamma correction parameter.
    impl->m_gamma_correct = m_params.strings().exist("gamma_correction");
    impl->m_target_gamma =
        impl->m_gamma_correct
            ? m_params.get_required<float>("gamma_correction", 2.2f)
            : 1.0f;
    impl->m_rcp_target_gamma = 1.0f / impl->m_target_gamma;
}

Color4f Frame::linear_rgb_to_frame(const Color4f& linear_rgb) const
{
    Color4f result;

    // Transform the input color to the color space of the frame.
    switch (impl->m_color_space)
    {
      case ColorSpaceLinearRGB:
        result = linear_rgb;
        break;

      case ColorSpaceSRGB:
        result.rgb() = fast_linear_rgb_to_srgb(linear_rgb.rgb());
        result.a = linear_rgb.a;
        break;

      case ColorSpaceCIEXYZ:
        result.rgb() = linear_rgb_to_ciexyz(linear_rgb.rgb());
        result.a = linear_rgb.a;
        break;

      default:
        assert(!"Invalid target color space.");
        result = linear_rgb;
        break;
    }

    // Clamp all pixel color channels to [0,1].
    // todo: mark clamped pixels in the diagnostic map.
    result = saturate(result);

    // Gamma-correct the pixel color if gamma correction is enabled.
    if (impl->m_gamma_correct)
    {
        // todo: investigate the usage of fast_pow() for gamma correction.
        const float rcp_target_gamma = impl->m_rcp_target_gamma;
        result[0] = pow(result[0], rcp_target_gamma);
        result[1] = pow(result[1], rcp_target_gamma);
        result[2] = pow(result[2], rcp_target_gamma);
    }

    return result;
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
