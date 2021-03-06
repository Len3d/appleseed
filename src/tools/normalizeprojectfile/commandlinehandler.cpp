
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
#include "commandlinehandler.h"

// appleseed.shared headers.
#include "application/superlogger.h"

// appleseed.foundation headers.
#include "foundation/utility/log.h"
#include "foundation/utility/string.h"

// Standard headers.
#include <cstdlib>

using namespace appleseed::shared;
using namespace foundation;
using namespace std;

namespace appleseed {
namespace normalizeprojectfile {

CommandLineHandler::CommandLineHandler()
  : CommandLineHandlerBase("normalizeprojectfile")
{
    m_filename.set_exact_value_count(1);
    parser().set_default_option_handler(&m_filename);
}

void CommandLineHandler::parse(
    const int       argc,
    const char*     argv[],
    SuperLogger&    logger)
{
    CommandLineHandlerBase::parse(argc, argv, logger);

    if (!m_filename.is_set())
        exit(0);
}

void CommandLineHandler::print_program_usage(
    const char*     program_name,
    SuperLogger&    logger) const
{
    LogTargetBase& log_target = logger.get_log_target();

    const int old_flags =
        log_target.set_formatting_flags(LogMessage::Info, LogMessage::DisplayMessage);

    LOG_INFO(logger, "usage: %s [options] project.appleseed", program_name);
    LOG_INFO(logger, "options:");

    parser().print_usage(logger);

    log_target.set_formatting_flags(LogMessage::Info, old_flags);
}

}   // namespace normalizeprojectfile
}   // namespace appleseed
