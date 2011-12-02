
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

#ifndef APPLESEED_STUDIO_MAINWINDOW_PROJECT_COLLECTIONITEMBASE_H
#define APPLESEED_STUDIO_MAINWINDOW_PROJECT_COLLECTIONITEMBASE_H

// appleseed.studio headers.
#include "mainwindow/project/itembase.h"

// appleseed.foundation headers.
#include "foundation/utility/foreach.h"
#include "foundation/utility/uid.h"

// Qt headers.
#include <QFont>
#include <QString>

// Standard headers.
#include <cassert>
#include <map>

namespace appleseed {
namespace studio {

template <typename Entity>
class CollectionItemBase
  : public virtual ItemBase
{
  public:
    CollectionItemBase();

    void add_item(Entity* entity);

    template <typename EntityContainer>
    void add_items(EntityContainer& items);

    void remove_item(const foundation::UniqueID entity_id);

  protected:
    typedef std::map<foundation::UniqueID, ItemBase*> ItemMap;

    ItemMap m_items;

    void add_item(const int index, Entity* entity);

    virtual ItemBase* create_item(Entity* entity) const;
};


//
// CollectionItemBase class implementation.
//

template <typename Entity>
CollectionItemBase<Entity>::CollectionItemBase()
{
    set_allow_edition(false);
    set_allow_deletion(false);

    QFont font;
    font.setBold(true);
    setFont(0, font);
}

template <typename Entity>
void CollectionItemBase<Entity>::add_item(Entity* entity)
{
    assert(entity);

    //
    // Insert the entity's item such that the list of items stays sorted.
    //
    // Unfortunately, QTreeWidgetItem::sortChildren() only works if a model
    // is bound to the tree view, so we can't use this method.
    //
    // Equally unfortunately, we can't easily use qLowerBound() either since
    // the QTreeWidgetItem class doesn't expose an iterator-based interface.
    //

    const QString entity_name(entity->get_name());
    int index = 0;

    if (childCount() > 0)
    {
        int end = childCount();

        while (end - index > 0)
        {
            const int middle = (index + end) / 2;

            if (QString::localeAwareCompare(child(middle)->text(0), entity_name) > 0)
                end = middle;
            else index = middle + 1;
        }
    }

    add_item(index, entity);
}

template <typename Entity>
template <typename EntityContainer>
void CollectionItemBase<Entity>::add_items(EntityContainer& entities)
{
    for (foundation::each<EntityContainer> i = entities; i; ++i)
        add_item(&*i);
}

template <typename Entity>
void CollectionItemBase<Entity>::remove_item(const foundation::UniqueID entity_id)
{
    const ItemMap::iterator it = m_items.find(entity_id);
    assert(it != m_items.end());

    removeChild(it->second);

    m_items.erase(it);
}

template <typename Entity>
void CollectionItemBase<Entity>::add_item(const int index, Entity* entity)
{
    assert(entity);

    ItemBase* item = create_item(entity);

    insertChild(index, item);

    m_items[entity->get_uid()] = item;
}

template <typename Entity>
ItemBase* CollectionItemBase<Entity>::create_item(Entity* entity) const
{
    assert(entity);

    return new ItemBase(entity->get_class_uid(), entity->get_name());
}

}       // namespace studio
}       // namespace appleseed

#endif  // !APPLESEED_STUDIO_MAINWINDOW_PROJECT_COLLECTIONITEMBASE_H
