
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

#ifndef APPLESEED_RENDERER_MODELING_INPUT_INPUTARRAY_H
#define APPLESEED_RENDERER_MODELING_INPUT_INPUTARRAY_H

// appleseed.foundation headers.
#include "foundation/core/concepts/noncopyable.h"
#include "foundation/math/vector.h"

// appleseed.main headers.
#include "main/dllsymbol.h"

// Standard headers.
#include <cstddef>

// Forward declarations.
namespace renderer      { class Source; }
namespace renderer      { class TextureCache; }

namespace renderer
{

//
// Input formats.
//

enum InputFormat
{
    InputFormatScalar = 0,
    InputFormatSpectrum
};


//
// Input array.
//

class DLLSYMBOL InputArray
  : public foundation::NonCopyable
{
  public:
    class iterator;

    // Constant iterator.
    class DLLSYMBOL const_iterator
    {
      public:
        // Value type.
        typedef const_iterator value_type;

        // Construction from a mutable iterator.
        const_iterator(const iterator& rhs);

        // Copy constructor.
        const_iterator(const const_iterator& rhs);

        // Assignment operator.
        const_iterator& operator=(const const_iterator& rhs);

        // Equality and inequality tests.
        bool operator==(const const_iterator& rhs) const;
        bool operator!=(const const_iterator& rhs) const;

        // Preincrement and predecrement operators.
        const_iterator& operator++();
        const_iterator& operator--();

        // Dereference operator.
        const const_iterator& operator*() const;

        // Get the name of this input.
        const char* name() const;

        // Get the format of this input.
        InputFormat format() const;

        // Get the default value of this input, or 0 if this input has no default value.
        const char* default_value() const;

        // Get the source bound to this input, or 0 if no source is bound to this input.
        Source* source() const;

      protected:
        friend class InputArray;

        // Constructor.
        const_iterator(const InputArray* array, const size_t index);

        const InputArray*   m_input_array;
        size_t              m_input_index;
    };

    // Mutable iterator.
    class DLLSYMBOL iterator
      : public const_iterator
    {
      public:
        // Value type.
        typedef iterator value_type;

        // Copy constructor.
        iterator(const iterator& rhs);

        // Assignment operator.
        iterator& operator=(const iterator& rhs);

        // Preincrement and predecrement operators.
        iterator& operator++();
        iterator& operator--();

        // Dereference operator.
        iterator& operator*();

        // Bind a source to this input, unbinding any previously bound source.
        // The ownership of the source is passed to renderer::InputArray.
        // 'source' may be 0, in which case the input is unbound.
        void bind(Source* source);

      private:
        friend class InputArray;

        // Constructor.
        iterator(const InputArray* array, const size_t index);
    };

    // Constructor.
    InputArray();

    // Destructor.
    ~InputArray();

    // Declare an input.
    void declare(
        const char*                 name,
        const InputFormat           format,
        const char*                 default_value = 0);

    // Return mutable begin and end input iterators.
    iterator begin();
    iterator end();

    // Return constant begin and end input iterators.
    const_iterator begin() const;
    const_iterator end() const;

    // Find a given input.
    // Return end() if the input could not be found.
    iterator find(const char* name);
    const_iterator find(const char* name) const;

    // Get the source bound to a given input.
    // Return 0 if the input could not be found, or no source is bound to it.
    Source* source(const char* name) const;

    // Compute the cumulated size in bytes of the input values.
    size_t compute_data_size() const;

    // Evaluate all inputs into a preallocated block of memory.
    // The address 'values + offset' must be 16-byte aligned.
    void evaluate(
        TextureCache&               texture_cache,
        const foundation::Vector2d& uv,
        void*                       values,
        const size_t                offset = 0) const;

    // Evaluate all uniform inputs into a preallocated block of memory.
    // The address 'values + offset' must be 16-byte aligned.
    void evaluate_uniforms(
        void*                       values,
        const size_t                offset = 0) const;

  private:
    struct Impl;
    Impl* impl;
};

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_MODELING_INPUT_INPUTARRAY_H
