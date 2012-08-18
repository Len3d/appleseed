
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

#ifndef APPLESEED_CLI_COMMANDLINEHANDLER_H
#define APPLESEED_CLI_COMMANDLINEHANDLER_H

// appleseed.foundation headers.
#include "foundation/utility/commandlineparser.h"

// appleseed.shared headers.
#include "application/commandlinehandlerbase.h"

// Standard headers.
#include <string>

// Forward declarations.
namespace appleseed { namespace shared { class SuperLogger; } }

namespace appleseed {
namespace cli {

//
// Command line handler.
//

class CommandLineHandler
  : public shared::CommandLineHandlerBase
{
  public:
    // General options.
#if defined __APPLE__ || defined _WIN32
    foundation::FlagOptionHandler                   m_display_output;
#endif
    foundation::ValueOptionHandler<std::string>     m_run_unit_tests;
    foundation::ValueOptionHandler<std::string>     m_run_unit_benchmarks;
    foundation::FlagOptionHandler                   m_verbose_unit_tests;
    foundation::FlagOptionHandler                   m_dump_entity_definitions;

    foundation::ValueOptionHandler<std::string>     m_configuration;
    foundation::ValueOptionHandler<std::string>     m_params;
    foundation::ValueOptionHandler<std::string>     m_filenames;

    foundation::FlagOptionHandler                   m_benchmark_mode;

    // Aliases for rendering options.
    foundation::ValueOptionHandler<int>             m_rendering_threads;
    foundation::ValueOptionHandler<std::string>     m_output;
    foundation::ValueOptionHandler<int>             m_resolution;
    foundation::ValueOptionHandler<int>             m_window;
    foundation::ValueOptionHandler<int>             m_samples;
    foundation::ValueOptionHandler<std::string>     m_override_shading;

    // Constructor.
    CommandLineHandler();

  private:
    // Emit usage instructions to the logger.
    virtual void print_program_usage(
        const char*             program_name,
        shared::SuperLogger&    logger) const;
};

}       // namespace cli
}       // namespace appleseed

#endif  // !APPLESEED_CLI_COMMANDLINEHANDLER_H
