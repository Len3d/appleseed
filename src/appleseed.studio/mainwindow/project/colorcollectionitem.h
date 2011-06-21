
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

#ifndef APPLESEED_STUDIO_MAINWINDOW_PROJECT_COLORCOLLECTIONITEM_H
#define APPLESEED_STUDIO_MAINWINDOW_PROJECT_COLORCOLLECTIONITEM_H

// appleseed.studio headers.
#include "mainwindow/project/collectionitem.h"

// appleseed.renderer headers.
#include "renderer/api/color.h"
#include "renderer/api/scene.h"

// Qt headers.
#include <QObject>

// Forward declarations.
namespace appleseed { namespace studio { class ProjectBuilder; } }

namespace appleseed {
namespace studio {

class SceneColorCollectionItem
  : public CollectionItem<renderer::ColorEntity, renderer::Scene>
{
    Q_OBJECT

  public:
    SceneColorCollectionItem(
        renderer::Scene&            scene,
        renderer::ColorContainer&   colors,
        ProjectBuilder&             project_builder);

    virtual void add_item(renderer::ColorEntity* color);

  private:
    renderer::Scene&        m_scene;
    ProjectBuilder&         m_project_builder;
};

class AssemblyColorCollectionItem
  : public CollectionItem<renderer::ColorEntity, renderer::Assembly>
{
    Q_OBJECT

  public:
    AssemblyColorCollectionItem(
        renderer::Assembly&         assembly,
        renderer::ColorContainer&   colors,
        ProjectBuilder&             project_builder);

    virtual void add_item(renderer::ColorEntity* color);

  private:
    renderer::Assembly&     m_assembly;
    ProjectBuilder&         m_project_builder;
};

}       // namespace studio
}       // namespace appleseed

#endif  // !APPLESEED_STUDIO_MAINWINDOW_PROJECT_COLORCOLLECTIONITEM_H
