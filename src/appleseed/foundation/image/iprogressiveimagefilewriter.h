
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

#ifndef APPLESEED_FOUNDATION_IMAGE_IPROGRESSIVEIMAGEFILEWRITER_H
#define APPLESEED_FOUNDATION_IMAGE_IPROGRESSIVEIMAGEFILEWRITER_H

// appleseed.foundation headers.
#include "foundation/core/concepts/noncopyable.h"
#include "foundation/image/imageattributes.h"

// Standard headers.
#include <cstddef>

// Forward declarations.
namespace foundation    { class CanvasProperties; }
namespace foundation    { class Tile; }

//
// On Windows, define FOUNDATIONDLL to __declspec(dllexport) when building the DLL
// and to __declspec(dllimport) when building an application using the DLL.
// Other platforms don't use this export mechanism and the symbol FOUNDATIONDLL is
// defined to evaluate to nothing.
//

#ifndef FOUNDATIONDLL
#ifdef _WIN32
#ifdef APPLESEED_FOUNDATION_EXPORTS
#define FOUNDATIONDLL __declspec(dllexport)
#else
#define FOUNDATIONDLL __declspec(dllimport)
#endif
#else
#define FOUNDATIONDLL
#endif
#endif

namespace foundation
{

//
// Progressive image file writer interface.
//

class FOUNDATIONDLL IProgressiveImageFileWriter
  : public NonCopyable
{
  public:
    // Destructor.
    virtual ~IProgressiveImageFileWriter() {}

    // Open an image file for writing.
    virtual void open(
        const char*             filename,
        const CanvasProperties& props,
        const ImageAttributes&  attrs = ImageAttributes()) = 0;

    // Close the image file.
    virtual void close() = 0;

    // Return true if an image file is currently open.
    virtual bool is_open() const = 0;

    // Write a tile to the image file.
    virtual void write_tile(
        const Tile&             tile,
        const size_t            tile_x,
        const size_t            tile_y) = 0;
};

}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_IMAGE_IPROGRESSIVEIMAGEFILEWRITER_H
