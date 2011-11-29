
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

#ifndef APPLESEED_FOUNDATION_MESH_OBJMESHFILEREADER_H
#define APPLESEED_FOUNDATION_MESH_OBJMESHFILEREADER_H

// appleseed.foundation headers.
#include "foundation/core/exceptions/exception.h"
#include "foundation/mesh/imeshfilereader.h"

// Standard headers.
#include <cstddef>
#include <string>

// Forward declarations.
namespace foundation    { class IMeshBuilder; }
namespace foundation    { class IOBJMeshBuilder; }

namespace foundation
{

//
// Wavefront OBJ mesh file reader.
//
// Reference:
//
//   http://people.scs.fsu.edu/~burkardt/txt/obj_format.txt
//

class OBJMeshFileReader
  : public IMeshFileReader
{
  public:
    // Parse error exception.
    struct ExceptionParseError
      : public Exception
    {
        const size_t m_line;                    // the line at which the parse error occurred
        explicit ExceptionParseError(const size_t line)
          : m_line(line)
        {
        }
    };

    // Exception thrown when an invalid face definition is encountered.
    struct ExceptionInvalidFaceDef
      : public ExceptionParseError
    {
        explicit ExceptionInvalidFaceDef(const size_t line)
          : ExceptionParseError(line)
        {
        }
    };

    // Reading options.
    enum Options
    {
        Defaults                = 0,
        StopOnInvalidFaceDef    = 1 << 0        // stop parsing on invalid face definitions
    };

    // Constructor.
    explicit OBJMeshFileReader(const Options options = Defaults);

    // Destructor.
    virtual ~OBJMeshFileReader();

    // Read an OBJ mesh file.
    virtual void read(
        const std::string&  filename,
        IMeshBuilder&       builder);
    void read(
        const std::string&  filename,
        IOBJMeshBuilder&    builder);

  private:
    // Private implementation.
    struct Impl;
    Impl* impl;
};

}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_MESH_OBJMESHFILEREADER_H
