
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
#include "glrenderwidget.h"

// appleseed.foundation headers.
#include "foundation/core/exceptions/exceptionnotimplemented.h"

// Qt headers.
#include <Qt>

// Standard headers.
#include <cassert>

using namespace foundation;
using namespace renderer;

namespace appleseed {
namespace studio {

//
// GLRenderWidget class implementation.
//

GLRenderWidget::GLRenderWidget(
    const int       width,
    const int       height,
    QWidget*        parent)
  : QGLWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);

    setFixedWidth(width);
    setFixedHeight(height);

    setAutoFillBackground(false);
    setAttribute(Qt::WA_OpaquePaintEvent, true);

//  setAttribute(Qt::WA_NoSystemBackground, true);
//  setAttribute(Qt::WA_PaintOnScreen, true);
}

void GLRenderWidget::clear(const Color4f& color)
{
    glClearColor(color[0], color[1], color[2], color[3]);
    glClear(GL_COLOR_BUFFER_BIT);
}

void GLRenderWidget::highlight_region(
    const size_t    x,
    const size_t    y,
    const size_t    width,
    const size_t    height)
{
    throw ExceptionNotImplemented();
}

void GLRenderWidget::blit_tile(
    const Frame&    frame,
    const size_t    tile_x,
    const size_t    tile_y)
{
    throw ExceptionNotImplemented();
}

void GLRenderWidget::blit_frame(
    const Frame&    frame)
{
    throw ExceptionNotImplemented();
}

void GLRenderWidget::initializeGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void GLRenderWidget::resizeGL(int width, int height)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0);

    glMatrixMode(GL_MODELVIEW);
}

void GLRenderWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
}

}   // namespace studio
}   // namespace appleseed
