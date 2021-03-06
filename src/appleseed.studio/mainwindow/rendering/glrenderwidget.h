
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

#ifndef APPLESEED_STUDIO_MAINWINDOW_RENDERING_GLRENDERWIDGET_H
#define APPLESEED_STUDIO_MAINWINDOW_RENDERING_GLRENDERWIDGET_H

// appleseed.foundation headers.
#include "foundation/image/color.h"

// appleseed.studio headers.
#include "mainwindow/rendering/irenderwidget.h"

// Qt headers.
#include <QGLWidget>

// Standard headers.
#include <cstddef>

// Forward declarations.
namespace renderer      { class Frame; }
class QWidget;

namespace appleseed {
namespace studio {

//
// An OpenGL-based render widget.
//

class GLRenderWidget
  : public QGLWidget
  , public IRenderWidget
{
  public:
    // Constructor.
    GLRenderWidget(
        const int                   width,
        const int                   height,
        QWidget*                    parent = 0);

    virtual void clear(
        const foundation::Color4f&  color);

    virtual void highlight_region(
        const size_t                x,
        const size_t                y,
        const size_t                width,
        const size_t                height);

    virtual void blit_tile(
        const renderer::Frame&      frame,
        const size_t                tile_x,
        const size_t                tile_y);

    virtual void blit_frame(
        const renderer::Frame&      frame);

  private:
    virtual void initializeGL();
    virtual void resizeGL(int width, int height);
    virtual void paintGL();
};

}       // namespace studio
}       // namespace appleseed

#endif  // !APPLESEED_STUDIO_MAINWINDOW_RENDERING_GLRENDERWIDGET_H
