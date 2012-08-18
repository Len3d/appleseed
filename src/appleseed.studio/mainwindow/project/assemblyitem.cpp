
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
#include "assemblyitem.h"

// appleseed.studio headers.
#include "mainwindow/project/collectionitem.h"
#include "mainwindow/project/multimodelcollectionitem.h"
#include "mainwindow/project/objectcollectionitem.h"
#include "mainwindow/project/objectinstancecollectionitem.h"
#include "mainwindow/project/projectbuilder.h"
#include "mainwindow/project/singlemodelcollectionitem.h"
#include "mainwindow/project/texturecollectionitem.h"
#include "mainwindow/project/tools.h"

// appleseed.renderer headers.
#include "renderer/api/bsdf.h"
#include "renderer/api/edf.h"
#include "renderer/api/entity.h"
#include "renderer/api/light.h"
#include "renderer/api/material.h"
#include "renderer/api/scene.h"
#include "renderer/api/surfaceshader.h"

// appleseed.foundation headers.
#include "foundation/utility/uid.h"

// Qt headers.
#include <QMenu>
#include <QMessageBox>
#include <QString>

// Standard headers.
#include <string>

using namespace foundation;
using namespace renderer;
using namespace std;

namespace appleseed {
namespace studio {

namespace
{
    const UniqueID g_class_uid = new_guid();
}

AssemblyItem::AssemblyItem(
    Assembly&       assembly,
    BaseGroup&      parent,
    BaseGroupItem*  parent_item,
    ProjectBuilder& project_builder,
    ParamArray&     settings)
  : BaseGroupItem(g_class_uid, assembly, project_builder, settings)
  , m_assembly(assembly)
  , m_parent(parent)
  , m_parent_item(parent_item)
  , m_project_builder(project_builder)
{
    set_title(QString::fromAscii(assembly.get_name()));
    set_render_layer(QString::fromAscii(assembly.get_render_layer_name()));

    set_allow_edition(false);

    insertChild(
        3,
        m_bsdf_collection_item = add_multi_model_collection_item<BSDF>(assembly.bsdfs()));

    insertChild(
        4,
        m_edf_collection_item = add_multi_model_collection_item<EDF>(assembly.edfs()));

    insertChild(
        5,
        m_surface_shader_collection_item = add_multi_model_collection_item<SurfaceShader>(assembly.surface_shaders()));

    insertChild(
        6,
        m_material_collection_item = add_single_model_collection_item<Material>(assembly.materials()));

    insertChild(
        7,
        m_light_collection_item = add_multi_model_collection_item<Light>(assembly.lights()));

    insertChild(
        8,
        m_object_collection_item =
            new ObjectCollectionItem(
                assembly.objects(),
                assembly,
                this,
                project_builder,
                settings));

    insertChild(
        9,
        m_object_instance_collection_item =
            new ObjectInstanceCollectionItem(
                assembly.object_instances(),
                assembly,
                this,
                project_builder));
}

QMenu* AssemblyItem::get_single_item_context_menu() const
{
    QMenu* menu = ItemBase::get_single_item_context_menu();

    menu->addSeparator();
    menu->addAction("Instantiate...", this, SLOT(slot_instantiate()));

    menu->addSeparator();
    menu->addAction("Import Objects...", m_object_collection_item, SLOT(slot_import_objects()));
    menu->addAction("Import Textures...", &get_texture_collection_item(), SLOT(slot_import_textures()));

    menu->addSeparator();
    menu->addAction("Create Assembly...", &get_assembly_collection_item(), SLOT(slot_create()));
    menu->addAction("Create BSDF...", m_bsdf_collection_item, SLOT(slot_create()));
    menu->addAction("Create Color...", &get_color_collection_item(), SLOT(slot_create()));
    menu->addAction("Create EDF...", m_edf_collection_item, SLOT(slot_create()));
    menu->addAction("Create Light...", m_light_collection_item, SLOT(slot_create()));
    menu->addAction("Create Material...", m_material_collection_item, SLOT(slot_create()));
    menu->addAction("Create Surface Shader...", m_surface_shader_collection_item, SLOT(slot_create()));

    return menu;
}

void AssemblyItem::add_item(BSDF* bsdf)
{
    m_bsdf_collection_item->add_item(bsdf);
}

void AssemblyItem::add_item(EDF* edf)
{
    m_edf_collection_item->add_item(edf);
}

void AssemblyItem::add_item(SurfaceShader* surface_shader)
{
    m_surface_shader_collection_item->add_item(surface_shader);
}

void AssemblyItem::add_item(Material* material)
{
    m_material_collection_item->add_item(material);
}

void AssemblyItem::add_item(Light* light)
{
    m_light_collection_item->add_item(light);
}

void AssemblyItem::add_item(Object* object)
{
    m_object_collection_item->add_item(object);
}

void AssemblyItem::add_item(ObjectInstance* object_instance)
{
    m_object_instance_collection_item->add_item(object_instance);
}

ObjectCollectionItem& AssemblyItem::get_object_collection_item() const
{
    return *m_object_collection_item;
}

ObjectInstanceCollectionItem& AssemblyItem::get_object_instance_collection_item() const
{
    return *m_object_instance_collection_item;
}

void AssemblyItem::slot_instantiate()
{
    const string instance_name_suggestion =
        get_name_suggestion(
            string(m_assembly.get_name()) + "_inst",
            m_parent.assembly_instances());

    const string instance_name =
        get_entity_name_dialog(
            treeWidget(),
            "Instantiate Assembly",
            "Assembly Instance Name:",
            instance_name_suggestion);

    if (!instance_name.empty())
    {
        m_project_builder.insert_assembly_instance(
            m_parent,
            m_parent_item,
            instance_name,
            m_assembly);
    }
}

template <typename Entity, typename EntityContainer>
CollectionItem<Entity, Assembly, AssemblyItem>* AssemblyItem::add_single_model_collection_item(EntityContainer& entities)
{
    CollectionItem<Entity, Assembly, AssemblyItem>* item =
        new SingleModelCollectionItem<Entity, Assembly, AssemblyItem>(
            new_guid(),
            EntityTraits<Entity>::get_human_readable_collection_type_name(),
            m_assembly,
            this,
            m_project_builder);

    item->add_items(entities);

    return item;
}

template <typename Entity, typename EntityContainer>
CollectionItem<Entity, Assembly, AssemblyItem>* AssemblyItem::add_multi_model_collection_item(EntityContainer& entities)
{
    CollectionItem<Entity, Assembly, AssemblyItem>* item =
        new MultiModelCollectionItem<Entity, Assembly, AssemblyItem>(
            new_guid(),
            EntityTraits<Entity>::get_human_readable_collection_type_name(),
            m_assembly,
            this,
            m_project_builder);

    item->add_items(entities);

    return item;
}

namespace
{
    int ask_assembly_deletion_confirmation(const char* assembly_name)
    {
        QMessageBox msgbox;
        msgbox.setWindowTitle("Delete Assembly?");
        msgbox.setIcon(QMessageBox::Question);
        msgbox.setText(QString("You are about to delete the assembly \"%1\" and all its instances.").arg(assembly_name));
        msgbox.setInformativeText("Continue?");
        msgbox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgbox.setDefaultButton(QMessageBox::No);
        return msgbox.exec();
    }
}

void AssemblyItem::slot_delete()
{
    if (!allows_deletion())
        return;

    const char* assembly_name = m_assembly.get_name();

    if (ask_assembly_deletion_confirmation(assembly_name) != QMessageBox::Yes)
        return;

    m_project_builder.remove_assembly(
        m_parent,
        m_parent_item,
        m_assembly.get_uid());

    // 'this' no longer exists at this point.
}

}   // namespace studio
}   // namespace appleseed
