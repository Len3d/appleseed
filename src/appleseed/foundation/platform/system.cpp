
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
#include "system.h"

// appleseed.foundation headers.
#include "foundation/platform/thread.h"
#include "foundation/platform/x86timer.h"

// Windows.
#if defined _WIN32

    // appleseed.foundation headers.
    #include "foundation/platform/windows.h"

    // Standard headers.
    #include <cassert>
    #include <cstdlib>

// Mac OS X.
#elif defined __APPLE__

    // Platform headers.
    #include <sys/sysctl.h>

// Linux.
#elif defined __linux__

    // Platform headers.
    #include <unistd.h>

// Unsupported platform.
#else
#error Unsupported platform.
#endif

namespace foundation
{

// ------------------------------------------------------------------------------------------------
// Common code.
// ------------------------------------------------------------------------------------------------

size_t System::get_logical_cpu_core_count()
{
    const size_t concurrency =
        static_cast<size_t>(boost::thread::hardware_concurrency());

    return concurrency > 1 ? concurrency : 1;
}

uint64 System::get_cpu_core_frequency(const uint32 calibration_time_ms)
{
    return X86Timer(calibration_time_ms).frequency();
}

// ------------------------------------------------------------------------------------------------
// Windows.
// ------------------------------------------------------------------------------------------------

#if defined _WIN32

namespace
{
    //
    // This code is based on a code snippet by Nick Strupat (http://stackoverflow.com/a/4049562).
    //

    bool get_cache_descriptor(const size_t level, CACHE_DESCRIPTOR& result)
    {
        assert(level >= 1 && level <= 3);

        DWORD buffer_size = 0;
        BOOL success = GetLogicalProcessorInformation(0, &buffer_size);

        if (success == TRUE || GetLastError() != ERROR_INSUFFICIENT_BUFFER)
            return false;

        SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)malloc(buffer_size);
        success = GetLogicalProcessorInformation(buffer, &buffer_size);

        if (success == FALSE)
            return false;

        bool found = false;

        for (size_t i = 0; i < buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); ++i)
        {
            if (buffer[i].Relationship == RelationCache)
            {
                const CACHE_DESCRIPTOR& cache = buffer[i].Cache;

                if (cache.Level == level && (cache.Type == CacheData || cache.Type == CacheUnified))
                {
                    found = true;
                    result = cache;
                    break;
                }
            }
        }

        free(buffer);

        return found;
    }
}

size_t System::get_l1_data_cache_size()
{
    CACHE_DESCRIPTOR cache;
    return get_cache_descriptor(1, cache) ? cache.Size : 0;
}

size_t System::get_l1_data_cache_line_size()
{
    CACHE_DESCRIPTOR cache;
    return get_cache_descriptor(1, cache) ? cache.LineSize : 0;
}

size_t System::get_l2_cache_size()
{
    CACHE_DESCRIPTOR cache;
    return get_cache_descriptor(2, cache) ? cache.Size : 0;
}

size_t System::get_l2_cache_line_size()
{
    CACHE_DESCRIPTOR cache;
    return get_cache_descriptor(2, cache) ? cache.LineSize : 0;
}

size_t System::get_l3_cache_size()
{
    CACHE_DESCRIPTOR cache;
    return get_cache_descriptor(3, cache) ? cache.Size : 0;
}

size_t System::get_l3_cache_line_size()
{
    CACHE_DESCRIPTOR cache;
    return get_cache_descriptor(3, cache) ? cache.LineSize : 0;
}

// ------------------------------------------------------------------------------------------------
// Mac OS X.
// ------------------------------------------------------------------------------------------------

#elif defined __APPLE__

namespace
{
    size_t get_system_value(const char* name)
    {
        size_t value = 0;
        size_t value_size = sizeof(value);
        return sysctlbyname(name, &value, &value_size, 0, 0) == 0 ? value : 0;
    }
}

size_t System::get_l1_data_cache_size()
{
    return get_system_value("hw.l1dcachesize");
}

size_t System::get_l1_data_cache_line_size()
{
    return get_l1_data_cache_size() > 0 ? get_system_value("hw.cachelinesize") : 0;
}

size_t System::get_l2_cache_size()
{
    return get_system_value("hw.l2cachesize");
}

size_t System::get_l2_cache_line_size()
{
    return get_l2_cache_size() > 0 ? get_system_value("hw.cachelinesize") : 0;
}

size_t System::get_l3_cache_size()
{
    return get_system_value("hw.l3cachesize");
}

size_t System::get_l3_cache_line_size()
{
    return get_l3_cache_size() > 0 ? get_system_value("hw.cachelinesize") : 0;
}

// ------------------------------------------------------------------------------------------------
// Linux.
// ------------------------------------------------------------------------------------------------

#elif defined __linux__

size_t System::get_l1_data_cache_size()
{
    return sysconf(_SC_LEVEL1_DCACHE_SIZE);
}

size_t System::get_l1_data_cache_line_size()
{
    return sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
}

size_t System::get_l2_cache_size()
{
    return sysconf(_SC_LEVEL2_CACHE_SIZE);
}

size_t System::get_l2_cache_line_size()
{
    return sysconf(_SC_LEVEL2_CACHE_LINESIZE);
}

size_t System::get_l3_cache_size()
{
    return sysconf(_SC_LEVEL3_CACHE_SIZE);
}

size_t System::get_l3_cache_line_size()
{
    return sysconf(_SC_LEVEL3_CACHE_LINESIZE);
}

#endif

}   // namespace foundation
