
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

// Interface header.
#include "assemblyitem.h"

// appleseed.studio headers.
#include "mainwindow/project/collectionitem.h"
#include "mainwindow/project/colorcollectionitem.h"
#include "mainwindow/project/itemtypemap.h"
#include "mainwindow/project/lightcollectionitem.h"
#include "mainwindow/project/multimodelcollectionitem.h"
#include "mainwindow/project/objectcollectionitem.h"
#include "mainwindow/project/objectinstancecollectionitem.h"
#include "mainwindow/project/projectbuilder.h"
#include "mainwindow/project/projecttree.h"
#include "mainwindow/project/singlemodelcollectionitem.h"
#include "mainwindow/project/texturecollectionitem.h"
#include "mainwindow/project/textureinstancecollectionitem.h"
#include "mainwindow/project/tools.h"

// appleseed.renderer headers.
#include "renderer/api/bsdf.h"
#include "renderer/api/color.h"
#include "renderer/api/edf.h"
#include "renderer/api/entity.h"
#include "renderer/api/light.h"
#include "renderer/api/material.h"
#include "renderer/api/object.h"
#include "renderer/api/scene.h"
#include "renderer/api/surfaceshader.h"
#include "renderer/api/texture.h"

// appleseed.foundation headers.
#include "foundation/utility/uid.h"

// Qt headers.
#include <QFileDialog>
#include <QMenu>
#include <QString>
#include <QStringList>

// Standard headers.
#include <string>

using namespace foundation;
using namespace renderer;
using namespace std;

namespace appleseed {
namespace studio {

struct AssemblyItem::Impl
{
    AssemblyItem*                               m_assembly_item;
    Scene&                                      m_scene;
    ProjectBuilder&                             m_project_builder;
    Assembly&                                   m_assembly;

    AssemblyColorCollectionItem*                m_color_collection_item;
    TextureCollectionItem*                      m_texture_collection_item;
    TextureInstanceCollectionItem*              m_texture_instance_collection_item;
    CollectionItem<BSDF, Assembly>*             m_bsdf_collection_item;
    CollectionItem<EDF, Assembly>*              m_edf_collection_item;
    CollectionItem<SurfaceShader, Assembly>*    m_surface_shader_collection_item;
    CollectionItem<Material, Assembly>*         m_material_collection_item;
    LightCollectionItem*                        m_light_collection_item;
    ObjectCollectionItem*                       m_object_collection_item;
    ObjectInstanceCollectionItem*               m_object_instance_collection_item;

    Impl(
        AssemblyItem*   assembly_item,
        Scene&          scene,
        Assembly&       assembly,
        ProjectBuilder& project_builder)
      : m_assembly_item(assembly_item)
      , m_scene(scene)
      , m_assembly(assembly)
      , m_project_builder(project_builder)
    {
        m_color_collection_item = add_color_collection_item(assembly.colors());
        m_texture_collection_item = add_collection_item(assembly.textures());
        m_texture_instance_collection_item = add_collection_item(assembly.texture_instances());
        m_bsdf_collection_item = add_multi_model_collection_item<BSDF>(assembly.bsdfs());
        m_edf_collection_item = add_multi_model_collection_item<EDF>(assembly.edfs());
        m_surface_shader_collection_item = add_multi_model_collection_item<SurfaceShader>(assembly.surface_shaders());
        m_material_collection_item = add_single_model_collection_item<Material>(assembly.materials());
        m_light_collection_item = add_collection_item(assembly.lights());
        m_object_collection_item = add_collection_item(assembly.objects());
        m_object_instance_collection_item = add_collection_item(assembly.object_instances());
    }

    template <typename EntityContainer>
    typename ItemTypeMap<EntityContainer>::T* add_collection_item(EntityContainer& entities)
    {
        typedef typename ItemTypeMap<EntityContainer>::T ItemType;

        ItemType* item =
            new ItemType(
                m_assembly,
                entities,
                m_project_builder);

        m_assembly_item->addChild(item);

        return item;
    }

    template <typename Entity, typename EntityContainer>
    CollectionItem<Entity, Assembly>* add_single_model_collection_item(EntityContainer& entities)
    {
        CollectionItem<Entity, Assembly>* item =
            new SingleModelCollectionItem<Entity, Assembly>(
                new_guid(),
                EntityTraits<Entity>::get_human_readable_collection_type_name(),
                m_assembly,
                m_project_builder);

        item->add_items(entities);

        m_assembly_item->addChild(item);

        return item;
    }

    template <typename Entity, typename EntityContainer>
    CollectionItem<Entity, Assembly>* add_multi_model_collection_item(EntityContainer& entities)
    {
        CollectionItem<Entity, Assembly>* item =
            new MultiModelCollectionItem<Entity, Assembly>(
                new_guid(),
                EntityTraits<Entity>::get_human_readable_collection_type_name(),
                m_assembly,
                m_project_builder);

        item->add_items(entities);

        m_assembly_item->addChild(item);

        return item;
    }

    AssemblyColorCollectionItem* add_color_collection_item(ColorContainer& colors)
    {
        AssemblyColorCollectionItem* item =
            new AssemblyColorCollectionItem(
                m_assembly,
                colors,
                m_project_builder);

        m_assembly_item->addChild(item);

        return item;
    }
};

AssemblyItem::AssemblyItem(
    Scene&          scene,
    Assembly&       assembly,
    ProjectBuilder& project_builder)
  : ItemBase(assembly.get_class_uid(), assembly.get_name())
  , impl(new Impl(this, scene, assembly, project_builder))
{
}

AssemblyItem::~AssemblyItem()
{
    delete impl;
}

QMenu* AssemblyItem::get_single_item_context_menu() const
{
    QMenu* menu = ItemBase::get_single_item_context_menu();
    menu->addSeparator();

    menu->addAction("Instantiate...", this, SLOT(slot_instantiate()));
    menu->addSeparator();

    menu->addAction("Import Objects...", impl->m_object_collection_item, SLOT(slot_import_objects()));
    menu->addAction("Import Textures...", impl->m_texture_collection_item, SLOT(slot_import_textures()));
    menu->addSeparator();

    menu->addAction("Create BSDF...", impl->m_bsdf_collection_item, SLOT(slot_create()));
    menu->addAction("Create EDF...", impl->m_edf_collection_item, SLOT(slot_create()));
    menu->addAction("Create Surface Shader...", impl->m_surface_shader_collection_item, SLOT(slot_create()));
    menu->addAction("Create Material...", impl->m_material_collection_item, SLOT(slot_create()));

    return menu;
}

void AssemblyItem::add_item(ColorEntity* color)
{
    impl->m_color_collection_item->add_item(color);
}

void AssemblyItem::add_item(Texture* texture)
{
    impl->m_texture_collection_item->add_item(texture);
}

void AssemblyItem::add_item(TextureInstance* texture_instance)
{
    impl->m_texture_instance_collection_item->add_item(texture_instance);
}

void AssemblyItem::add_item(BSDF* bsdf)
{
    impl->m_bsdf_collection_item->add_item(bsdf);
}

void AssemblyItem::add_item(EDF* edf)
{
    impl->m_edf_collection_item->add_item(edf);
}

void AssemblyItem::add_item(SurfaceShader* surface_shader)
{
    impl->m_surface_shader_collection_item->add_item(surface_shader);
}

void AssemblyItem::add_item(Material* material)
{
    impl->m_material_collection_item->add_item(material);
}

void AssemblyItem::add_item(Light* light)
{
    impl->m_light_collection_item->add_item(light);
}

void AssemblyItem::add_item(Object* object)
{
    impl->m_object_collection_item->add_item(object);
}

void AssemblyItem::add_item(ObjectInstance* object_instance)
{
    impl->m_object_instance_collection_item->add_item(object_instance);
}

void AssemblyItem::slot_instantiate()
{
    const string instance_name_suggestion =
        get_name_suggestion(
            string(impl->m_assembly.get_name()) + "_inst",
            impl->m_scene.assembly_instances());

    const string instance_name =
        get_entity_name_dialog(
            treeWidget(),
            "Instantiate Assembly",
            "Assembly Instance Name:",
            instance_name_suggestion);

    if (!instance_name.empty())
        impl->m_project_builder.insert_assembly_instance(instance_name, impl->m_assembly);
}

}   // namespace studio
}   // namespace appleseed
