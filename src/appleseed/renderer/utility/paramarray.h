
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

#ifndef APPLESEED_RENDERER_UTILITY_PARAMARRAY_H
#define APPLESEED_RENDERER_UTILITY_PARAMARRAY_H

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/utility/paramarray.h"

// appleseed.foundation headers.
#include "foundation/utility/containers/dictionary.h"
#include "foundation/utility/string.h"

// appleseed.main headers.
#include "main/dllsymbol.h"

// Standard headers.
#include <cassert>
#include <string>

namespace renderer
{

//
// A collection of parameters.
//

class DLLSYMBOL ParamArray
  : public foundation::Dictionary
{
  public:
    // Constructors.
    ParamArray();
    ParamArray(const ParamArray& rhs);
    ParamArray(const foundation::Dictionary& dictionary);

    // Assignment operator.
    ParamArray& operator=(const ParamArray& rhs);

    // Insert an item into the dictionary.
    template <typename T> ParamArray& insert(const char* key, const T& value);

    //
    // Retrieve the value of a required parameter.
    //
    // If the parameter is present but its value is invalid, an error message
    // is emitted and the default value is returned.
    //
    // If the parameter is missing, an error message is emitted and the default
    // value is returned.
    //

    template <typename T>
    T get_required(const char* name, const T& default_value) const;

    //
    // Retrieve the value of an optional parameter.
    //
    // If the parameter is present but its value is invalid, an error message
    // is emitted and the default value is returned.
    //
    // If the parameter is missing, the default value is silently returned.
    //

    template <typename T>
    T get_optional(const char* name, const T& default_value = T()) const;

    //
    // Insert an item through a given hierarchy, creating branches as needed.
    //
    // For instance, insert_path("a.b.c", 12) will insert the value 12 with key 'c'
    // inside a dictionary named 'b' itself contained inside a dictionary named 'a'.
    //

    ParamArray& insert_path(const char* path, const char* value);
    template <typename T> ParamArray& insert_path(const char* path, const T& value);
    template <typename T> ParamArray& insert_path(const std::string& path, const T& value);

    // Retrieve an item in a given hierarchy.
    const char* get_path(const char* path) const;

    // Like get_required() but given a path instead of a key.
    template <typename T>
    T get_path_required(const char* path, const T& default_value) const;

    // Like get_optional() but given a path instead of a key.
    template <typename T>
    T get_path_optional(const char* path, const T& default_value = T()) const;

    // Return a child set of parameters, or create it if it doesn't exist.
    ParamArray& push(const char* name);

    // Retrieve a child set of parameters.
    // Returns an empty parameter set if the specified set cannot be found.
    const ParamArray& child(const char* name) const;

    // Merge another set of parameters into this one.
    ParamArray& merge(const ParamArray& rhs);

  private:
    template <typename T>
    T get_helper(
        const char*     name,
        const T&        default_value,
        const bool      required) const;

    template <typename T>
    T get_path_helper(
        const char*     path,
        const T&        default_value,
        const bool      required) const;
};


//
// ParamArray class implementation.
//

template <typename T>
inline ParamArray& ParamArray::insert(const char* key, const T& value)
{
    foundation::Dictionary::insert(key, value);
    return *this;
}

template <>
inline ParamArray& ParamArray::insert(const char* key, const ParamArray& value)
{
    dictionaries().insert(key, value);
    return *this;
}

template <typename T>
inline T ParamArray::get_required(const char* name, const T& default_value) const
{
    return get_helper(name, default_value, true);
}

template <typename T>
inline T ParamArray::get_optional(const char* name, const T& default_value) const
{
    return get_helper(name, default_value, false);
}

template <typename T>
inline ParamArray& ParamArray::insert_path(const char* path, const T& value)
{
    return insert_path(path, foundation::to_string(value).c_str());
}

template <typename T>
inline ParamArray& ParamArray::insert_path(const std::string& path, const T& value)
{
    return insert_path(path.c_str(), value);
}

template <typename T>
inline T ParamArray::get_path_required(const char* path, const T& default_value) const
{
    return get_path_helper(path, default_value, true);
}

template <typename T>
inline T ParamArray::get_path_optional(const char* path, const T& default_value) const
{
    return get_path_helper(path, default_value, false);
}

template <typename T>
T ParamArray::get_helper(
    const char*     name,
    const T&        default_value,
    const bool      required) const
{
    assert(name);

    try
    {
        return get<T>(name);
    }
    catch (const foundation::ExceptionDictionaryItemNotFound&)
    {
        if (required)
        {
            RENDERER_LOG_ERROR(
                "parameter \"%s\" not found, using default value \"%s\".",
                name,
                foundation::to_string(default_value).c_str());
        }
    }
    catch (const foundation::ExceptionStringConversionError&)
    {
        RENDERER_LOG_ERROR(
            "invalid value \"%s\" for parameter \"%s\", using default value \"%s\".",
            get(name),
            name,
            foundation::to_string(default_value).c_str());
    }

    return default_value;
}

template <typename T>
T ParamArray::get_path_helper(
    const char*     path,
    const T&        default_value,
    const bool      required) const
{
    assert(path);

    try
    {
        return foundation::from_string<T>(get_path(path));
    }
    catch (const foundation::ExceptionDictionaryItemNotFound&)
    {
        if (required)
        {
            RENDERER_LOG_ERROR(
                "parameter \"%s\" not found, using default value \"%s\".",
                path,
                foundation::to_string(default_value).c_str());
        }
    }
    catch (const foundation::ExceptionStringConversionError&)
    {
        RENDERER_LOG_ERROR(
            "invalid value \"%s\" for parameter \"%s\", using default value \"%s\".",
            get_path(path),
            path,
            foundation::to_string(default_value).c_str());
    }

    return default_value;
}

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_UTILITY_PARAMARRAY_H
