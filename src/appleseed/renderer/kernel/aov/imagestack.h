
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

#ifndef APPLESEED_RENDERER_KERNEL_AOV_IMAGESTACK_H
#define APPLESEED_RENDERER_KERNEL_AOV_IMAGESTACK_H

// appleseed.foundation headers.
#include "foundation/image/pixel.h"

// appleseed.main headers.
#include "main/dllsymbol.h"

// Standard headers.
#include <cstddef>

// Forward declarations.
namespace foundation    { class Image; }
namespace renderer      { class TileStack; }

namespace renderer
{

//
// A stack of named images.
//

class DLLSYMBOL ImageStack
{
  public:
    ImageStack(
        const size_t                    canvas_width,
        const size_t                    canvas_height,
        const size_t                    tile_width,
        const size_t                    tile_height);

    ~ImageStack();

    bool empty() const;

    size_t size() const;

    const char* get_name(const size_t index) const;

    const foundation::Image& get_image(const size_t index) const;

    void clear();

    void append(
        const char*                     name,
        const foundation::PixelFormat   format);

    size_t get_or_append(
        const char*                     name,
        const foundation::PixelFormat   format);

    TileStack tiles(
        const size_t                    tile_x,
        const size_t                    tile_y) const;

  private:
    struct Impl;
    Impl* impl;
};

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_KERNEL_AOV_IMAGESTACK_H
