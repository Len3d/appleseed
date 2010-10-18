
//
// This source file is part of appleseed.
// Visit http://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2010 Francois Beaune
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
#include "surfaceshadercollectionitem.h"

// appleseed.studio headers.
#include "mainwindow/project/assemblyentitybrowser.h"
#include "mainwindow/project/entityeditorformfactory.h"
#include "mainwindow/project/entityeditorwindow.h"
#include "mainwindow/project/entityitem.h"
#include "mainwindow/project/projectbuilder.h"
#include "mainwindow/project/tools.h"

// appleseed.renderer headers.
#include "renderer/api/surfaceshader.h"

// Qt headers.
#include <QMenu>

// Standard headers.
#include <memory>

using namespace foundation;
using namespace renderer;
using namespace std;

namespace appleseed {
namespace studio {

SurfaceShaderCollectionItem::SurfaceShaderCollectionItem(
    Assembly&               assembly,
    SurfaceShaderContainer& surface_shaders,
    ProjectBuilder&         project_builder)
  : CollectionItem("Surface Shaders", surface_shaders)
  , m_assembly(assembly)
  , m_project_builder(project_builder)
{
}

QMenu* SurfaceShaderCollectionItem::get_context_menu() const
{
    QMenu* menu = new QMenu(treeWidget());
    menu->addAction("Create Surface Shader...", this, SLOT(slot_create_surface_shader()));
    return menu;
}

void SurfaceShaderCollectionItem::slot_create_surface_shader()
{
    auto_ptr<EntityEditorWindow::IFormFactory> form_factory(
        new EntityEditorFormFactory<SurfaceShaderFactoryRegistrar>(
            m_registrar,
            get_name_suggestion("surface_shader", m_assembly.surface_shaders())));

    auto_ptr<EntityEditorWindow::IEntityBrowser> entity_browser(
        new AssemblyEntityBrowser(m_assembly));

    open_entity_editor(
        treeWidget(),
        "Create Surface Shader",
        form_factory,
        entity_browser,
        this,
        SLOT(slot_create_surface_shader_accepted(foundation::Dictionary)));
}

void SurfaceShaderCollectionItem::slot_create_surface_shader_accepted(Dictionary values)
{
    m_project_builder.insert_surface_shader(m_assembly, values);

    qobject_cast<QWidget*>(sender())->close();
}

}   // namespace studio
}   // namespace appleseed