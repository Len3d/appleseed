
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

#ifndef APPLESEED_FOUNDATION_UTILITY_CONTAINERS_SPECIALIZEDARRAYS_H
#define APPLESEED_FOUNDATION_UTILITY_CONTAINERS_SPECIALIZEDARRAYS_H

// appleseed.foundation headers.
#include "foundation/utility/containers/array.h"
#include "foundation/utility/containers/dictionary.h"

// Standard headers.
#include <cstddef>

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
// Predefined array types.
//

DECLARE_ARRAY(FloatArray, float);
DECLARE_ARRAY(DoubleArray, double);
DECLARE_ARRAY(DictionaryArray, Dictionary);


//
// An array of strings that can be passed safely across DLL boundaries.
//

class FOUNDATIONDLL StringArray
{
  public:
    // Types.
    typedef const char* value_type;
    typedef size_t size_type;

    // Constructors.
    StringArray();
    StringArray(const StringArray& rhs);
    StringArray(
        const size_type     size,
        const value_type*   values);

    // Destructor.
    ~StringArray();

    // Assignment operator.
    StringArray& operator=(const StringArray& rhs);

    // Returns the size of the vector.
    size_type size() const;

    // Tests if the vector is empty.
    bool empty() const;

    // Clears the vector.
    void clear();

    // Reserves memory for a given number of elements.
    void reserve(const size_type count);

    // Specifies a new size for a vector.
    void resize(const size_type new_size);

    // Adds an element to the end of the vector.
    void push_back(const value_type val);

    // Set the vector element at a specified position.
    void set(const size_type pos, const value_type val);

    // Returns the vector element at a specified position.
    value_type operator[](const size_type pos) const;

  private:
    struct Impl;
    Impl* impl;
};

}       // namespace foundation

#endif  // !APPLESEED_FOUNDATION_UTILITY_CONTAINERS_SPECIALIZEDARRAYS_H
