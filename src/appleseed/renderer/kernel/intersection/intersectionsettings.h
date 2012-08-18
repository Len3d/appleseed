
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

#ifndef APPLESEED_RENDERER_KERNEL_INTERSECTION_INTERSECTIONSETTINGS_H
#define APPLESEED_RENDERER_KERNEL_INTERSECTION_INTERSECTIONSETTINGS_H

// appleseed.renderer headers.
#include "renderer/global/globaltypes.h"

// appleseed.foundation headers.
#include "foundation/math/intersection.h"

// Standard headers.
#include <cstddef>

namespace renderer
{

//
// Triangle types.
//

// Triangle format used for storage.
typedef foundation::TriangleMT<GScalar> GTriangleType;

// Triangle format used for intersection.
typedef foundation::TriangleMT<double> TriangleType;
typedef foundation::TriangleMTSupportPlane<double> TriangleSupportPlaneType;


//
// Assembly tree settings.
//

// Maximum number of assemblies per leaf.
const size_t AssemblyTreeMaxLeafSize = 1;

// Relative cost of traversing an interior node.
const double AssemblyTreeInteriorNodeTraversalCost = 1.0;

// Relative cost of intersecting an assembly.
const double AssemblyTreeTriangleIntersectionCost = 10.0;


//
// Region tree settings.
//

// Maximum region duplication rate.
const double RegionTreeMaxDuplication = 2.0;

// Maximum number of regions per leaf.
const size_t RegionTreeMaxLeafSize = 64;

// Maximum depth of the tree.
const size_t RegionTreeMaxDepth = 16;

// Size of the region tree access cache.
const size_t RegionTreeAccessCacheSize = 16;


//
// Triangle tree settings.
//

// Maximum number of triangles per leaf.
const size_t TriangleTreeMaxLeafSize = 2;

// Relative cost of traversing an interior node.
const GScalar TriangleTreeInteriorNodeTraversalCost(1.0);

// Relative cost of intersecting a triangle.
const GScalar TriangleTreeTriangleIntersectionCost(1.0);

// Number of bins used during SBVH construction.
const size_t TriangleTreeBinCount = 256;

// Define this symbol to enable reordering the nodes of triangle trees for better
// locality of reference. Requires a lot of temporary memory for minimal results.
#undef RENDERER_TRIANGLE_TREE_REORDER_NODES

// Depth of a subtree in the van Emde Boas node layout.
const size_t TriangleTreeSubtreeDepth = 2;

// Size of the triangle tree access cache.
const size_t TriangleTreeAccessCacheSize = 16;

// Size of the stack (in number of nodes) used during traversal.
const size_t TriangleTreeStackSize = 64;


//
// Miscellaneous settings.
//

// If defined, an adaptive procedure is used to offset intersection points.
// If left undefined, a fixed, constant-time procedure is used. The adaptive
// procedure handles degenerate cases better but is slightly slower. It must
// be used when the triangle model is set to Moller-Trumbore (MT).
#define RENDERER_ADAPTIVE_OFFSET

}       // namespace renderer

#endif  // !APPLESEED_RENDERER_KERNEL_INTERSECTION_INTERSECTIONSETTINGS_H
