
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
#include "texturecollectionitem.h"

// appleseed.studio headers.
#include "mainwindow/project/projectbuilder.h"
#include "mainwindow/project/textureitem.h"
#include "utility/interop.h"
#include "utility/settingskeys.h"

// appleseed.renderer headers.
#include "renderer/api/utility.h"

// appleseed.foundation headers.
#include "foundation/utility/uid.h"

// Qt headers.
#include <QDir>
#include <QFileDialog>
#include <QMenu>
#include <QString>
#include <QStringList>

// boost headers.
#include "boost/filesystem/path.hpp"

// Standard headers.
#include <string>

using namespace boost;
using namespace foundation;
using namespace renderer;
using namespace std;

namespace appleseed {
namespace studio {

namespace
{
    const UniqueID g_class_uid = new_guid();
}

TextureCollectionItem::TextureCollectionItem(
    TextureContainer&   textures,
    BaseGroup&          parent,
    BaseGroupItem*      parent_item,
    ProjectBuilder&     project_builder,
    ParamArray&         settings)
  : CollectionItemBase<Texture>(g_class_uid, "Textures")
  , m_parent(parent)
  , m_parent_item(parent_item)
  , m_project_builder(project_builder)
  , m_settings(settings)
{
    add_items(textures);
}

QMenu* TextureCollectionItem::get_single_item_context_menu() const
{
    QMenu* menu = CollectionItemBase<Texture>::get_single_item_context_menu();

    menu->addSeparator();
    menu->addAction("Import Textures...", this, SLOT(slot_import_textures()));

    return menu;
}

void TextureCollectionItem::slot_import_textures()
{
    QFileDialog::Options options;
    QString selected_filter;

    const QStringList filepaths =
        QFileDialog::getOpenFileNames(
            treeWidget(),
            "Import Textures...",
            m_settings.get_path_optional<QString>(LAST_DIRECTORY_SETTINGS_KEY),
            "Texture Files (*.exr);;All Files (*.*)",
            &selected_filter,
            options);

    if (filepaths.empty())
        return;

    const filesystem::path path(
        QDir::toNativeSeparators(filepaths.first()).toStdString());

    m_settings.insert_path(
        LAST_DIRECTORY_SETTINGS_KEY,
        path.parent_path().string());

    for (int i = 0; i < filepaths.size(); ++i)
    {
        const string filepath = QDir::toNativeSeparators(filepaths[i]).toStdString();
        m_project_builder.insert_texture(m_parent, m_parent_item, filepath);
    }
}

ItemBase* TextureCollectionItem::create_item(Texture* texture) const
{
    assert(texture);

    return new TextureItem(texture, m_parent, m_parent_item, m_project_builder);
}

}   // namespace studio
}   // namespace appleseed
