
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

// appleseed.foundation headers.
#include "foundation/utility/log.h"
#include "foundation/utility/test.h"

using namespace foundation;

TEST_SUITE(Foundation_Utility_Log_LogTargetBase)
{
    struct FakeLogTarget
      : public LogTargetBase
    {
        virtual void release()
        {
            delete this;
        }

        virtual void write(
            const LogMessage::Category  category,
            const char*                 file,
            const size_t                line,
            const char*                 message)
        {
        }
    };

    TEST_CASE_F(LogTargetBase_WhenConstructed_WarningCategoryHasDefaultFormattingFlags, FakeLogTarget)
    {
        EXPECT_EQ(
            LogMessage::DefaultFormattingFlags,
            get_formatting_flags(LogMessage::Warning));
    }

    TEST_CASE_F(GetFormattingFlags_GivenSetWarningCategoryFlags_ReturnsThoseFlags, FakeLogTarget)
    {
        const int ExpectedFlags = LogMessage::DisplayDate | LogMessage::DisplayLine;

        set_formatting_flags(LogMessage::Warning, ExpectedFlags);

        EXPECT_EQ(
            ExpectedFlags,
            get_formatting_flags(LogMessage::Warning));
    }

    TEST_CASE_F(HasFormattingFlags_GivenSetWarningCategoryFlags_ReturnsTrueForSetFlags, FakeLogTarget)
    {
        const int ExpectedFlags = LogMessage::DisplayDate | LogMessage::DisplayLine;

        set_formatting_flags(LogMessage::Warning, ExpectedFlags);

        EXPECT_TRUE(has_formatting_flags(LogMessage::Warning, LogMessage::DisplayDate));
        EXPECT_TRUE(has_formatting_flags(LogMessage::Warning, LogMessage::DisplayLine));
    }

    TEST_CASE_F(HasFormattingFlags_GivenUnsetWarningCategoryFlags_ReturnsFalseForUnsetFlags, FakeLogTarget)
    {
        const int ExpectedFlags = LogMessage::DisplayDate | LogMessage::DisplayLine;

        set_formatting_flags(LogMessage::Warning, ExpectedFlags);

        EXPECT_FALSE(has_formatting_flags(LogMessage::Warning, LogMessage::DisplayMessage));
    }
}
