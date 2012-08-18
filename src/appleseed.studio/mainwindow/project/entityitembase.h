
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

#ifndef APPLESEED_STUDIO_MAINWINDOW_PROJECT_ENTITYITEMBASE_H
#define APPLESEED_STUDIO_MAINWINDOW_PROJECT_ENTITYITEMBASE_H

// appleseed.studio headers.
#include "mainwindow/project/itembase.h"

// appleseed.foundation headers.
#include "foundation/utility/containers/dictionary.h"
#include "foundation/utility/uid.h"

// Qt headers.
#include <QObject>
#include <QString>

namespace appleseed {
namespace studio {

// Work around a limitation in Qt: a template class cannot have slots.
class EntityItemBaseSlots
  : public ItemBase
{
    Q_OBJECT

  public:
    explicit EntityItemBaseSlots(const foundation::UniqueID class_uid);
    EntityItemBaseSlots(const foundation::UniqueID class_uid, const QString& title);

  protected slots:
    virtual void slot_edit_accepted(foundation::Dictionary values);
};

template <typename Entity>
class EntityItemBase
  : public EntityItemBaseSlots
{
  public:
    explicit EntityItemBase(Entity* entity);

    void update();

  protected:
    Entity* m_entity;
};


//
// EntityItemBase class implementation.
//

template <typename Entity>
EntityItemBase<Entity>::EntityItemBase(Entity* entity)
  : EntityItemBaseSlots(entity->get_class_uid())
  , m_entity(entity)
{
    update();
}

template <typename Entity>
void EntityItemBase<Entity>::update()
{
    set_title(QString::fromAscii(m_entity->get_name()));
    set_render_layer(QString::fromAscii(m_entity->get_render_layer_name()));
}

}       // namespace studio
}       // namespace appleseed

#endif  // !APPLESEED_STUDIO_MAINWINDOW_PROJECT_ENTITYITEMBASE_H
