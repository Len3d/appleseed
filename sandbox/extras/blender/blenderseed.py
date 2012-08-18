
#
# This source file is part of appleseed.
# Visit http://appleseedhq.net/ for additional information and resources.
#
# This software is released under the MIT license.
#
# Copyright (c) 2010-2012 Francois Beaune, Jupiter Jazz Limited
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

# Imports.
import bpy
import cProfile
import dis
from datetime import datetime
import math
import mathutils
import os


#--------------------------------------------------------------------------------------------------
# Add-on information.
#--------------------------------------------------------------------------------------------------

bl_info = {
    "name": "appleseed project format",
    "description": "Exports a scene to the appleseed project file format.",
    "author": "Franz Beaune",
    "version": (1, 3, 2),
    "blender": (2, 6, 2),   # we really need Blender 2.62 or newer
    "api": 36339,
    "location": "File > Export",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export"}

script_name = "blenderseed.py"

def get_version_string():
    return "version " + ".".join(map(str, bl_info["version"]))


#--------------------------------------------------------------------------------------------------
# Settings.
#--------------------------------------------------------------------------------------------------

Verbose = False
EnableProfiling = False


#--------------------------------------------------------------------------------------------------
# Generic utilities.
#--------------------------------------------------------------------------------------------------

def square(x):
    return x * x

def rad_to_deg(rad):
    return rad * 180.0 / math.pi

def is_black(color):
    return color[0] == 0.0 and color[1] == 0.0 and color[2] == 0.0

def add(color1, color2):
    return [ color1[0] + color2[0], color1[1] + color2[1], color1[2] + color2[2] ]

def mul(color, multiplier):
    return [ color[0] * multiplier, color[1] * multiplier, color[2] * multiplier ]

def scene_enumerator(self, context):
    matches = []
    for scene in bpy.data.scenes:
        matches.append((scene.name, scene.name, ""))
    return matches

def camera_enumerator(self, context):
    return object_enumerator('CAMERA')

def object_enumerator(type):
    matches = []
    for object in bpy.data.objects:
        if object.type == type:
            matches.append((object.name, object.name, ""))
    return matches


#--------------------------------------------------------------------------------------------------
# Material-related utilities.
#--------------------------------------------------------------------------------------------------

class MatUtils:
    @staticmethod
    def compute_reflection_factor(material):
        return material.raytrace_mirror.reflect_factor if material.raytrace_mirror.use else 0.0

    @staticmethod
    def is_material_reflective(material):
        return MatUtils.compute_reflection_factor(material) > 0.0

    @staticmethod
    def compute_transparency_factor(material):
        material_transp_factor = (1.0 - material.alpha) if material.use_transparency else 0.0
        # We don't really support the Fresnel parameter yet, hack something together...
        material_transp_factor += material.raytrace_transparency.fresnel / 3.0
        return material_transp_factor

    @staticmethod
    def is_material_transparent(material):
        return MatUtils.compute_transparency_factor(material) > 0.0


#--------------------------------------------------------------------------------------------------
# Write a mesh object to disk in Wavefront OBJ format.
#--------------------------------------------------------------------------------------------------

def get_array2_key(v):
    return int(v[0] * 1000000), int(v[1] * 1000000)

def get_vector2_key(v):
    w = v * 1000000
    return int(w.x), int(w.y)

def get_vector3_key(v):
    w = v * 1000000
    return int(w.x), int(w.y), int(w.z)

def write_mesh_to_disk(mesh, mesh_faces, mesh_uvtex, filepath):
    with open(filepath, "w") as output_file:
        # Write file header.
        output_file.write("# File generated by %s %s.\n" % (script_name, get_version_string()))

        vertices = mesh.vertices
        faces = mesh_faces
        uvtex = mesh_uvtex
        uvset = uvtex.active.data if uvtex else None

        # Sort the faces by material.
        sorted_faces = [ (index, face) for index, face in enumerate(faces) ]
        sorted_faces.sort(key = lambda item: item[1].material_index)

        # Write vertices.
        output_file.write("# %d vertices.\n" % len(vertices))
        for vertex in vertices:
            v = vertex.co
            output_file.write("v %.15f %.15f %.15f\n" % (v.x, v.y, v.z))

        # Deduplicate and write normals.
        output_file.write("# Vertex normals.\n")
        normal_indices = {}
        vertex_normal_indices = {}
        face_normal_indices = {}
        current_normal_index = 0
        for face_index, face in sorted_faces:
            if face.use_smooth:
                for vertex_index in face.vertices:
                    vn = vertices[vertex_index].normal
                    vn_key = get_vector3_key(vn)
                    if vn_key in normal_indices:
                        vertex_normal_indices[vertex_index] = normal_indices[vn_key]
                    else:
                        output_file.write("vn %.15f %.15f %.15f\n" % (vn.x, vn.y, vn.z))
                        normal_indices[vn_key] = current_normal_index
                        vertex_normal_indices[vertex_index] = current_normal_index
                        current_normal_index += 1
            else:
                vn = face.normal
                vn_key = get_vector3_key(vn)
                if vn_key in normal_indices:
                    face_normal_indices[face_index] = normal_indices[vn_key]
                else:
                    output_file.write("vn %.15f %.15f %.15f\n" % (vn.x, vn.y, vn.z))
                    normal_indices[vn_key] = current_normal_index
                    face_normal_indices[face_index] = current_normal_index
                    current_normal_index += 1

        # Deduplicate and write texture coordinates.
        if uvset:
            output_file.write("# Texture coordinates.\n")
            vt_indices = {}
            vertex_texcoord_indices = {}
            current_vt_index = 0
            for face_index, face in sorted_faces:
                assert len(uvset[face_index].uv) == len(face.vertices)
                for vt_index, vt in enumerate(uvset[face_index].uv):
                    vertex_index = face.vertices[vt_index]
                    vt_key = get_array2_key(vt)
                    if vt_key in vt_indices:
                        vertex_texcoord_indices[face_index, vertex_index] = vt_indices[vt_key]
                    else:
                        output_file.write("vt %.15f %.15f\n" % (vt[0], vt[1]))
                        vt_indices[vt_key] = current_vt_index
                        vertex_texcoord_indices[face_index, vertex_index] = current_vt_index
                        current_vt_index += 1

        mesh_parts = []

        # Write faces.
        output_file.write("# %d faces.\n" % len(sorted_faces))
        current_material_index = -1
        for face_index, face in sorted_faces:
            if current_material_index != face.material_index:
                current_material_index = face.material_index
                mesh_name = "part_%d" % current_material_index
                mesh_parts.append((current_material_index, mesh_name))
                output_file.write("o {0}\n".format(mesh_name))
            line = "f"
            if uvset and len(uvset[face_index].uv) > 0:
                if face.use_smooth:
                    for vertex_index in face.vertices:
                        texcoord_index = vertex_texcoord_indices[face_index, vertex_index]
                        normal_index = vertex_normal_indices[vertex_index]
                        line += " %d/%d/%d" % (vertex_index + 1, texcoord_index + 1, normal_index + 1)
                else:
                    normal_index = face_normal_indices[face_index]
                    for vertex_index in face.vertices:
                        texcoord_index = vertex_texcoord_indices[face_index, vertex_index]
                        line += " %d/%d/%d" % (vertex_index + 1, texcoord_index + 1, normal_index + 1)
            else:
                if face.use_smooth:
                    for vertex_index in face.vertices:
                        normal_index = vertex_normal_indices[vertex_index]
                        line += " %d//%d" % (vertex_index + 1, normal_index + 1)
                else:
                    normal_index = face_normal_indices[face_index]
                    for vertex_index in face.vertices:
                        line += " %d//%d" % (vertex_index + 1, normal_index + 1)
            output_file.write(line + "\n")

        return mesh_parts


#--------------------------------------------------------------------------------------------------
# AppleseedExportOperator class.
#--------------------------------------------------------------------------------------------------

class AppleseedExportOperator(bpy.types.Operator):
    bl_idname = "appleseed.export"
    bl_label = "Export"

    # The name of the appleseed project file.
    filepath = bpy.props.StringProperty(subtype='FILE_PATH')

    selected_scene = bpy.props.EnumProperty(name="Scene",
                                            description="Select the scene to export",
                                            items=scene_enumerator)

    selected_camera = bpy.props.EnumProperty(name="Camera",
                                             description="Select the camera to export",
                                             items=camera_enumerator)

    lighting_engine = bpy.props.EnumProperty(name="Lighting Engine",
                                             description="Select the lighting engine to use",
                                             items=[('pt', "Path Tracing", "Full Global Illumination"),
                                                    ('drt', "Distributed Ray Tracing", "Direct Lighting Only")],
                                             default='pt')

    sample_count = bpy.props.IntProperty(name="Sample Count",
                                         description="Number of samples per pixels in final frame mode",
                                         min=1,
                                         max=1000000,
                                         default=25,
                                         subtype='UNSIGNED')

    export_emitting_obj_as_lights = bpy.props.BoolProperty(name="Export Emitting Objects As Mesh Lights",
                                                           description="Export object with light-emitting materials as mesh (area) lights",
                                                           default=False)

    point_lights_exitance_mult = bpy.props.FloatProperty(name="Point Lights Energy Multiplier",
                                                         description="Multiply the exitance of point lights by this factor",
                                                         min=0.0,
                                                         max=1000.0,
                                                         default=1.0,
                                                         subtype='FACTOR')

    spot_lights_exitance_mult = bpy.props.FloatProperty(name="Spot Lights Energy Multiplier",
                                                        description="Multiply the exitance of spot lights by this factor",
                                                        min=0.0,
                                                        max=1000.0,
                                                        default=1.0,
                                                        subtype='FACTOR')

    light_mats_exitance_mult = bpy.props.FloatProperty(name="Light-Emitting Materials Energy Multiplier",
                                                       description="Multiply the exitance of light-emitting materials by this factor",
                                                       min=0.0,
                                                       max=1000.0,
                                                       default=1.0,
                                                       subtype='FACTOR')

    env_exitance_mult = bpy.props.FloatProperty(name="Environment Energy Multiplier",
                                                description="Multiply the exitance of the environment by this factor",
                                                min=0.0,
                                                max=1000.0,
                                                default=1.0,
                                                subtype='FACTOR')

    specular_mult = bpy.props.FloatProperty(name="Specular Components Multiplier",
                                            description="Multiply the intensity of specular components by this factor",
                                            min=0.0,
                                            max=1000.0,
                                            default=1.0,
                                            subtype='FACTOR')

    enable_ibl = bpy.props.BoolProperty(name="Enable Image Based Lighting",
                                        description="If checked, Image Based Lighting (IBL) will be enabled during rendering",
                                        default=True)

    enable_caustics = bpy.props.BoolProperty(name="Enable Caustics (Path Tracing Only)",
                                             description="If checked, caustics will be enabled during rendering",
                                             default=True)

    generate_mesh_files = bpy.props.BoolProperty(name="Write Meshes to Disk",
                                                 description="If unchecked, the mesh files (.obj files) won't be regenerated",
                                                 default=True)

    recompute_vertex_normals = bpy.props.BoolProperty(name="Recompute Vertex Normals",
                                                      description="If checked, vertex normals will be recomputed during tessellation",
                                                      default=True)

    apply_modifiers = bpy.props.BoolProperty(name="Apply Modifiers",
                                             description="If checked, modifiers will be applied to objects during tessellation",
                                             default=True)

    tessellation_quality = bpy.props.EnumProperty(name="Tessellation Quality",
                                                  description="Fineness of the tessellation of non-mesh objects",
                                                  items=[('PREVIEW', "Preview", ""), ('RENDER', "Render", "")],
                                                  default='RENDER')

    # Transformation matrix applied to all entities of the scene.
    global_scale = 0.1
    global_matrix = mathutils.Matrix.Scale(global_scale, 4)

    def execute(self, context):
        if EnableProfiling:
            dis.dis(get_vector3_key)
            cProfile.runctx("self.export()", globals(), locals())
        else: self.export()
        return { 'FINISHED' }

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return { 'RUNNING_MODAL' }

    def __get_selected_scene(self):
        if self.selected_scene is not None and self.selected_scene in bpy.data.scenes:
            return bpy.data.scenes[self.selected_scene]
        else: return None

    def __get_selected_camera(self):
        if self.selected_camera is not None and self.selected_camera in bpy.data.objects:
            return bpy.data.objects[self.selected_camera]
        else: return None

    def export(self):
        scene = self.__get_selected_scene()

        if scene is None:
            self.__error("No scene to export.")
            return

        # Blender material -> front material name, back material name.
        self._emitted_materials = {}

        # Object name -> instance count.
        self._instance_count = {}

        # Object name -> (material index, mesh name).
        self._mesh_parts = {}

        file_path = os.path.splitext(self.filepath)[0] + ".appleseed"

        self.__info("")
        self.__info("Starting export of scene '{0}' to {1}...".format(scene.name, file_path))

        start_time = datetime.now()

        try:
            with open(file_path, "w") as self._output_file:
                self._indent = 0
                self.__emit_file_header()
                self.__emit_project(scene)
        except IOError:
            self.__error("Could not write to {0}.".format(file_path))
            return

        elapsed_time = datetime.now() - start_time
        self.__info("Finished exporting in {0}".format(elapsed_time))

    def __emit_file_header(self):
        self.__emit_line("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        self.__emit_line("<!-- File generated by {0} {1}. -->".format(script_name, get_version_string()))

    def __emit_project(self, scene):
        self.__open_element("project")
        self.__emit_scene(scene)
        self.__emit_output(scene)
        self.__emit_configurations()
        self.__close_element("project")

    #----------------------------------------------------------------------------------------------
    # Scene.
    #----------------------------------------------------------------------------------------------

    def __emit_scene(self, scene):
        self.__open_element("scene")
        self.__emit_camera(scene)
        self.__emit_environment(scene)
        self.__emit_assembly(scene)
        self.__emit_assembly_instance_element(scene)
        self.__close_element("scene")

    def __emit_assembly(self, scene):
        self.__open_element('assembly name="' + scene.name + '"')
        self.__emit_physical_surface_shader_element()
        self.__emit_default_material()
        self.__emit_objects(scene)
        self.__close_element("assembly")

    def __emit_assembly_instance_element(self, scene):
        self.__open_element('assembly_instance name="' + scene.name + '_instance" assembly="' + scene.name + '"')
        self.__close_element("assembly_instance")

    def __emit_objects(self, scene):
        for object in scene.objects:
            # Skip objects marked as non-renderable.
            if object.hide_render:
                if Verbose:
                    self.__info("Skipping object '{0}' because it is marked as non-renderable.".format(object.name))
                continue

            # Skip cameras since they are exported separately.
            if object.type == 'CAMERA':
                if Verbose:
                    self.__info("Skipping object '{0}' because its type is '{1}'.".format(object.name, object.type))
                continue

            if object.type == 'LAMP':
                self.__emit_light(scene, object)
            else:
                self.__emit_geometric_object(scene, object)

    #----------------------------------------------------------------------------------------------
    # Camera.
    #----------------------------------------------------------------------------------------------

    def __emit_camera(self, scene):
        camera = self.__get_selected_camera()

        if camera is None:
            self.__warning("No camera in the scene, exporting a default camera.")
            self.__emit_default_camera_element()
            return

        render = scene.render

        film_width = 32.0 / 1000                                # Blender's film width is hardcoded to 32 mm
        aspect_ratio = self.__get_frame_aspect_ratio(render)
        focal_length = camera.data.lens / 1000.0                # Blender's camera focal length is expressed in mm

        camera_matrix = self.global_matrix * camera.matrix_world
        origin = camera_matrix.col[3]
        forward = -camera_matrix.col[2]
        up = camera_matrix.col[1]
        target = origin + forward

        self.__open_element('camera name="' + camera.name + '" model="pinhole_camera"')
        self.__emit_parameter("film_width", film_width)
        self.__emit_parameter("aspect_ratio", aspect_ratio)
        self.__emit_parameter("focal_length", focal_length)
        self.__open_element("transform")
        self.__emit_line('<look_at origin="{0} {1} {2}" target="{3} {4} {5}" up="{6} {7} {8}" />'.format( \
                         origin[0], origin[2], -origin[1],
                         target[0], target[2], -target[1],
                         up[0], up[2], -up[1]))
        self.__close_element("transform")
        self.__close_element("camera")

    def __emit_default_camera_element(self):
        self.__open_element('camera name="camera" model="pinhole_camera"')
        self.__emit_parameter("film_width", 0.024892)
        self.__emit_parameter("film_height", 0.018669)
        self.__emit_parameter("focal_length", 0.035)
        self.__close_element("camera")
        return

    #----------------------------------------------------------------------------------------------
    # Environment.
    #----------------------------------------------------------------------------------------------

    def __emit_environment(self, scene):    
        horizon_exitance = [ 0.0, 0.0, 0.0 ]
        zenith_exitance = [ 0.0, 0.0, 0.0 ]

        # Add the contribution of the first hemi light found in the scene.
        found_hemi_light = False
        for object in scene.objects:
            if object.hide_render:
                continue
            if object.type == 'LAMP' and object.data.type == 'HEMI':
                if not found_hemi_light:
                    self.__info("Using hemi light '{0}' for environment lighting.".format(object.name))
                    hemi_exitance = mul(object.data.color, object.data.energy)
                    horizon_exitance = add(horizon_exitance, hemi_exitance)
                    zenith_exitance = add(zenith_exitance, hemi_exitance)
                    found_hemi_light = True
                else:
                    self.__warning("Ignoring hemi light '{0}', multiple hemi lights are not supported yet.".format(object.name))

        # Add the contribution of the sky.
        if scene.world is not None:
            horizon_exitance = add(horizon_exitance, scene.world.horizon_color)
            zenith_exitance = add(zenith_exitance, scene.world.zenith_color)

        # Emith the environment EDF and environment shader if necessary.
        if is_black(horizon_exitance) and is_black(zenith_exitance):
            env_edf_name = ""
            env_shader_name = ""
        else:
            # Emit the exitances.
            self.__emit_solid_linear_rgb_color_element("horizon_exitance", horizon_exitance, self.env_exitance_mult)
            self.__emit_solid_linear_rgb_color_element("zenith_exitance", zenith_exitance, self.env_exitance_mult)

            # Emit the environment EDF.
            env_edf_name = "environment_edf"
            self.__open_element('environment_edf name="{0}" model="gradient_environment_edf"'.format(env_edf_name))
            self.__emit_parameter("horizon_exitance", "horizon_exitance")
            self.__emit_parameter("zenith_exitance", "zenith_exitance")
            self.__close_element('environment_edf')

            # Emit the environment shader.
            env_shader_name = "environment_shader"
            self.__open_element('environment_shader name="{0}" model="edf_environment_shader"'.format(env_shader_name))
            self.__emit_parameter("environment_edf", env_edf_name)
            self.__close_element('environment_shader')

        # Emit the environment element.
        self.__open_element('environment name="environment" model="generic_environment"')
        self.__emit_parameter("environment_edf", env_edf_name)
        self.__emit_parameter("environment_shader", env_shader_name)
        self.__close_element('environment')

    #----------------------------------------------------------------------------------------------
    # Geometry.
    #----------------------------------------------------------------------------------------------

    def __emit_geometric_object(self, scene, object):
        # Print some information about this object in verbose mode.
        if Verbose:
            if object.parent:
                self.__info("------ Object '{0}' (type '{1}') child of object '{2}' ------".format(object.name, object.type, object.parent.name))
            else: self.__info("------ Object '{0}' (type '{1}') ------".format(object.name, object.type))

        # Skip children of dupli objects.
        if object.parent and object.parent.dupli_type in { 'VERTS', 'FACES' }:      # todo: what about dupli type 'GROUP'?
            if Verbose:
                self.__info("Skipping object '{0}' because its parent ('{1}') has dupli type '{2}'.".format(object.name, object.parent.name, object.parent.dupli_type))
            return

        # Create dupli list and collect dupli objects.
        if Verbose:
            self.__info("Object '{0}' has dupli type '{1}'.".format(object.name, object.dupli_type))
        if object.dupli_type != 'NONE':
            object.dupli_list_create(scene)
            dupli_objects = [ (dupli.object, dupli.matrix) for dupli in object.dupli_list ]
            if Verbose:
                self.__info("Object '{0}' has {1} dupli objects.".format(object.name, len(dupli_objects)))
        else:
            dupli_objects = [ (object, object.matrix_world) ]

        # Emit the dupli objects.
        for dupli_object in dupli_objects:
            self.__emit_dupli_object(scene, dupli_object[0], dupli_object[1])

        # Clear dupli list.
        if object.dupli_type != 'NONE':
            object.dupli_list_clear()

    def __emit_dupli_object(self, scene, object, object_matrix):
        # Emit the object the first time it is encountered.
        if object.name in self._instance_count:
            if Verbose:
                self.__info("Skipping export of object '{0}' since it was already exported.".format(object.name))
        else:
            try:
                # Tessellate the object.
                mesh = object.to_mesh(scene, self.apply_modifiers, self.tessellation_quality)

                if hasattr(mesh, 'polygons'):
                    # Blender 2.63 and newer: handle BMesh.
                    mesh.calc_tessface()
                    mesh_faces = mesh.tessfaces
                    mesh_uvtex = mesh.tessface_uv_textures
                else:
                    # Blender 2.62 and older.
                    mesh_faces = mesh.faces
                    mesh_uvtex = mesh.uv_textures
 
                # Write the geometry to disk and emit a mesh object element.
                self._mesh_parts[object.name] = self.__emit_mesh_object(scene, object, mesh, mesh_faces, mesh_uvtex)

                # Delete the tessellation.
                bpy.data.meshes.remove(mesh)
            except RuntimeError:
                self.__info("Skipping object '{0}' of type '{1}' because it could not be converted to a mesh.".format(object.name, object.type))
                return

        # Emit the object instance.
        self.__emit_mesh_object_instance(object, object_matrix)

    def __emit_mesh_object(self, scene, object, mesh, mesh_faces, mesh_uvtex):
        if len(mesh_faces) == 0:
            self.__info("Skipping object '{0}' since it has no faces once converted to a mesh.".format(object.name))
            return []

        mesh_filename = object.name + ".obj"

        if self.generate_mesh_files:
            # Recalculate vertex normals.
            if self.recompute_vertex_normals:
                mesh.calc_normals()

            # Export the mesh to disk.
            self.__progress("Exporting object '{0}' to {1}...".format(object.name, mesh_filename))
            mesh_filepath = os.path.join(os.path.dirname(self.filepath), mesh_filename)
            try:
                mesh_parts = write_mesh_to_disk(mesh, mesh_faces, mesh_uvtex, mesh_filepath)
                if Verbose:
                    self.__info("Object '{0}' exported as {1} meshes.".format(object.name, len(mesh_parts)))
            except IOError:
                self.__error("While exporting object '{0}': could not write to {1}, skipping this object.".format(object.name, mesh_filepath))
                return []
        else:
            # Build a list of mesh parts just as if we had exported the mesh to disk.
            material_indices = set()
            for face in mesh_faces:
                material_indices.add(face.material_index)
            mesh_parts = map(lambda material_index : (material_index, "part_%d" % material_index), material_indices)

        # Emit object.
        self.__emit_object_element(object.name, mesh_filename)

        return mesh_parts

    def __emit_mesh_object_instance(self, object, object_matrix):
        # Emit BSDFs and materials if they are encountered for the first time.
        for material_slot_index, material_slot in enumerate(object.material_slots):
            material = material_slot.material
            if material is None:
                self.__warning("While exporting instance of object '{0}': material slot #{1} has no material.".format(object.name, material_slot_index))
                continue
            if material not in self._emitted_materials:
                self._emitted_materials[material] = self.__emit_material(material)

        # Figure out the instance number of this object.
        if object.name in self._instance_count:
            instance_index = self._instance_count[object.name] + 1
        else:
            instance_index = 0
        self._instance_count[object.name] = instance_index
        if Verbose:
            self.__info("This is instance #{0} of object '{1}', it has {2} material slot(s).".format(instance_index, object.name, len(object.material_slots)))

        # Emit object parts instances.
        for (material_index, mesh_name) in self._mesh_parts[object.name]:
            part_name = "{0}.{1}".format(object.name, mesh_name)
            instance_name = "{0}.instance_{1}".format(part_name, instance_index)
            front_material_name = "__default_material"
            back_material_name = "__default_material"
            if material_index < len(object.material_slots):
                material = object.material_slots[material_index].material
                if material:
                    front_material_name, back_material_name = self._emitted_materials[material]
            self.__emit_object_instance_element(part_name, instance_name, self.global_matrix * object_matrix, front_material_name, back_material_name)

    def __emit_object_element(self, object_name, mesh_filepath):
        self.__open_element('object name="' + object_name + '" model="mesh_object"')
        self.__emit_parameter("filename", mesh_filepath)
        self.__close_element("object")

    def __emit_object_instance_element(self, object_name, instance_name, instance_matrix, front_material_name, back_material_name):
        self.__open_element('object_instance name="{0}" object="{1}"'.format(instance_name, object_name))
        self.__emit_transform_element(instance_matrix)
        self.__emit_line('<assign_material slot="0" side="front" material="{0}" />'.format(front_material_name))
        self.__emit_line('<assign_material slot="0" side="back" material="{0}" />'.format(back_material_name))
        self.__close_element("object_instance")

    #----------------------------------------------------------------------------------------------
    # Materials.
    #----------------------------------------------------------------------------------------------

    def __is_light_emitting_material(self, material):
        if material.get('appleseed_arealight', False):
            return True;

        return material.emit > 0.0 and self.export_emitting_obj_as_lights

    def __emit_physical_surface_shader_element(self):
        self.__emit_line('<surface_shader name="physical_surface_shader" model="physical_surface_shader" />')

    def __emit_default_material(self):
        self.__emit_solid_linear_rgb_color_element("__default_material_bsdf_reflectance", [ 0.8 ], 1.0)

        self.__open_element('bsdf name="__default_material_bsdf" model="lambertian_brdf"')
        self.__emit_parameter("reflectance", "__default_material_bsdf_reflectance")
        self.__close_element("bsdf")

        self.__emit_material_element("__default_material", "__default_material_bsdf", "", "physical_surface_shader")

    def __emit_material(self, material):
        if Verbose:
            self.__info("Translating material '{0}'...".format(material.name))

        if MatUtils.is_material_transparent(material):
            front_material_name = material.name + "_front"
            back_material_name = material.name + "_back"
            self.__emit_front_material(material, front_material_name)
            self.__emit_back_material(material, back_material_name)
        else:
            front_material_name = material.name
            self.__emit_front_material(material, front_material_name)
            if self.__is_light_emitting_material(material):
                # Assign the default material to the back face if the front face emits light,
                # as we don't want mesh lights to emit from both faces.
                back_material_name = "__default_material"
            else: back_material_name = front_material_name

        return front_material_name, back_material_name

    def __emit_front_material(self, material, material_name):
        bsdf_name = self.__emit_front_material_bsdf_tree(material, material_name)

        if self.__is_light_emitting_material(material):
            edf_name = "{0}_edf".format(material_name)
            self.__emit_edf(material, edf_name)
        else: edf_name = ""

        self.__emit_material_element(material_name, bsdf_name, edf_name, "physical_surface_shader")

    def __emit_back_material(self, material, material_name):
        bsdf_name = self.__emit_back_material_bsdf_tree(material, material_name)
        self.__emit_material_element(material_name, bsdf_name, "", "physical_surface_shader")

    def __emit_front_material_bsdf_tree(self, material, material_name):
        bsdfs = []

        # Transparent component.
        material_transp_factor = MatUtils.compute_transparency_factor(material)
        if material_transp_factor > 0.0:
            transp_bsdf_name = "{0}|transparent".format(material_name)
            self.__emit_specular_btdf(material, transp_bsdf_name, 'front')
            bsdfs.append([ transp_bsdf_name, material_transp_factor ])

        # Mirror component.
        material_refl_factor = MatUtils.compute_reflection_factor(material)
        if material_refl_factor > 0.0:
            mirror_bsdf_name = "{0}|specular".format(material_name)
            self.__emit_specular_brdf(material, mirror_bsdf_name)
            bsdfs.append([ mirror_bsdf_name, material_refl_factor ])

        # Diffuse/glossy component.
        dg_bsdf_name = "{0}|diffuseglossy".format(material_name)
        if is_black(material.specular_color * material.specular_intensity):
            self.__emit_lambertian_brdf(material, dg_bsdf_name)
        else:
            self.__emit_ashikhmin_brdf(material, dg_bsdf_name)
        material_dg_factor = 1.0 - max(material_transp_factor, material_refl_factor)
        bsdfs.append([ dg_bsdf_name, material_dg_factor ])

        return self.__emit_bsdf_blend(bsdfs)

    def __emit_back_material_bsdf_tree(self, material, material_name):
        transp_bsdf_name = "{0}|transparent".format(material_name)
        self.__emit_specular_btdf(material, transp_bsdf_name, 'back')
        return transp_bsdf_name

    def __emit_bsdf_blend(self, bsdfs):
        assert len(bsdfs) > 0

        # Only one BSDF, no blending.
        if len(bsdfs) == 1:
            return bsdfs[0][0]

        # Normalize weights if necessary.
        total_weight = 0.0
        for bsdf in bsdfs:
            total_weight += bsdf[1]
        if total_weight > 1.0:
            for bsdf in bsdfs:
                bsdf[1] /= total_weight

        # The left branch is simply the first BSDF.
        bsdf0_name = bsdfs[0][0]
        bsdf0_weight = bsdfs[0][1]

        # The right branch is a blend of all the other BSDFs (recurse).
        bsdf1_name = self.__emit_bsdf_blend(bsdfs[1:])
        bsdf1_weight = 1.0 - bsdf0_weight

        # Blend the left and right branches together.
        mix_name = "{0}+{1}".format(bsdf0_name, bsdf1_name)
        self.__emit_bsdf_mix(mix_name, bsdf0_name, bsdf0_weight, bsdf1_name, bsdf1_weight)

        return mix_name

    def __emit_lambertian_brdf(self, material, bsdf_name):
        reflectance_name = "{0}_reflectance".format(bsdf_name)
        self.__emit_solid_linear_rgb_color_element(reflectance_name,
                                                   material.diffuse_color,
                                                   material.diffuse_intensity)

        self.__open_element('bsdf name="{0}" model="lambertian_brdf"'.format(bsdf_name))
        self.__emit_parameter("reflectance", reflectance_name)
        self.__close_element("bsdf")

    def __emit_ashikhmin_brdf(self, material, bsdf_name):
        diffuse_reflectance_name = "{0}_diffuse_reflectance".format(bsdf_name)
        glossy_reflectance_name = "{0}_glossy_reflectance".format(bsdf_name)
        self.__emit_solid_linear_rgb_color_element(diffuse_reflectance_name,
                                                   material.diffuse_color,
                                                   material.diffuse_intensity)
        self.__emit_solid_linear_rgb_color_element(glossy_reflectance_name,
                                                   material.specular_color,
                                                   material.specular_intensity * self.specular_mult)

        self.__open_element('bsdf name="{0}" model="ashikhmin_brdf"'.format(bsdf_name))
        self.__emit_parameter("diffuse_reflectance", diffuse_reflectance_name)
        self.__emit_parameter("glossy_reflectance", glossy_reflectance_name)
        self.__emit_parameter("shininess_u", material.specular_hardness)
        self.__emit_parameter("shininess_v", material.specular_hardness)
        self.__close_element("bsdf")

    def __emit_specular_brdf(self, material, bsdf_name):
        reflectance_name = "{0}_reflectance".format(bsdf_name)
        self.__emit_solid_linear_rgb_color_element(reflectance_name, material.mirror_color, 1.0)

        self.__open_element('bsdf name="{0}" model="specular_brdf"'.format(bsdf_name))
        self.__emit_parameter("reflectance", reflectance_name)
        self.__close_element("bsdf")

    def __emit_specular_btdf(self, material, bsdf_name, side):
        assert side == 'front' or side == 'back'

        reflectance_name = "{0}_reflectance".format(bsdf_name)
        self.__emit_solid_linear_rgb_color_element(reflectance_name, [ 1.0 ], 1.0)

        transmittance_name = "{0}_transmittance".format(bsdf_name)
        self.__emit_solid_linear_rgb_color_element(transmittance_name, [ 1.0 ], 1.0)

        if material.transparency_method == 'RAYTRACE':
            if side == 'front':
                from_ior = 1.0
                to_ior = material.raytrace_transparency.ior
            else:
                from_ior = material.raytrace_transparency.ior
                to_ior = 1.0
        else:
            from_ior = 1.0
            to_ior = 1.0

        self.__open_element('bsdf name="{0}" model="specular_btdf"'.format(bsdf_name))
        self.__emit_parameter("reflectance", reflectance_name)
        self.__emit_parameter("transmittance", transmittance_name)
        self.__emit_parameter("from_ior", from_ior)
        self.__emit_parameter("to_ior", to_ior)
        self.__close_element("bsdf")

    def __emit_bsdf_mix(self, bsdf_name, bsdf0_name, bsdf0_weight, bsdf1_name, bsdf1_weight):
        self.__open_element('bsdf name="{0}" model="bsdf_mix"'.format(bsdf_name))
        self.__emit_parameter("bsdf0", bsdf0_name)
        self.__emit_parameter("weight0", bsdf0_weight)
        self.__emit_parameter("bsdf1", bsdf1_name)
        self.__emit_parameter("weight1", bsdf1_weight)
        self.__close_element("bsdf")

    def __emit_edf(self, material, edf_name):
        self.__emit_diffuse_edf(material, edf_name)

    def __emit_diffuse_edf(self, material, edf_name):
        exitance_name = "{0}_exitance".format(edf_name)
        emit_factor = material.emit if material.emit > 0.0 else 1.0
        self.__emit_solid_linear_rgb_color_element(exitance_name,
                                                   material.diffuse_color,
                                                   emit_factor * self.light_mats_exitance_mult)
        self.__emit_diffuse_edf_element(edf_name, exitance_name)

    def __emit_diffuse_edf_element(self, edf_name, exitance_name):
        self.__open_element('edf name="{0}" model="diffuse_edf"'.format(edf_name))
        self.__emit_parameter("exitance", exitance_name)
        self.__close_element("edf")

    def __emit_material_element(self, material_name, bsdf_name, edf_name, surface_shader_name):
        self.__open_element('material name="{0}" model="generic_material"'.format(material_name))
        if len(bsdf_name) > 0:
            self.__emit_parameter("bsdf", bsdf_name)
        if len(edf_name) > 0:
            self.__emit_parameter("edf", edf_name)
        self.__emit_parameter("surface_shader", surface_shader_name)
        self.__close_element("material")

    #----------------------------------------------------------------------------------------------
    # Lights.
    #----------------------------------------------------------------------------------------------

    def __emit_light(self, scene, object):
        light_type = object.data.type

        if light_type == 'POINT':
            self.__emit_point_light(scene, object)
        elif light_type == 'SPOT':
            self.__emit_spot_light(scene, object)
        elif light_type == 'HEMI':
            # Handle by the environment handling code.
            pass
        else:
            self.__warning("While exporting light '{0}': unsupported light type '{1}', skipping this light.".format(object.name, light_type))

    def __emit_point_light(self, scene, lamp):
        exitance_name = "{0}_exitance".format(lamp.name)
        self.__emit_solid_linear_rgb_color_element(exitance_name, lamp.data.color, lamp.data.energy * self.point_lights_exitance_mult)

        self.__open_element('light name="{0}" model="point_light"'.format(lamp.name))
        self.__emit_parameter("exitance", exitance_name)
        self.__emit_transform_element(self.global_matrix * lamp.matrix_world)
        self.__close_element("light")

    def __emit_spot_light(self, scene, lamp):
        exitance_name = "{0}_exitance".format(lamp.name)
        self.__emit_solid_linear_rgb_color_element(exitance_name, lamp.data.color, lamp.data.energy * self.spot_lights_exitance_mult)

        outer_angle = math.degrees(lamp.data.spot_size)
        inner_angle = (1.0 - lamp.data.spot_blend) * outer_angle

        self.__open_element('light name="{0}" model="spot_light"'.format(lamp.name))
        self.__emit_parameter("exitance", exitance_name)
        self.__emit_parameter("inner_angle", inner_angle)
        self.__emit_parameter("outer_angle", outer_angle)
        self.__emit_transform_element(self.global_matrix * lamp.matrix_world)
        self.__close_element("light")

    #----------------------------------------------------------------------------------------------
    # Output.
    #----------------------------------------------------------------------------------------------

    def __emit_output(self, scene):
        self.__open_element("output")
        self.__emit_frame_element(scene)
        self.__close_element("output")

    def __emit_frame_element(self, scene):
        camera = self.__get_selected_camera()
        width, height = self.__get_frame_resolution(scene.render)
        self.__open_element("frame name=\"beauty\"")
        self.__emit_parameter("camera", "camera" if camera is None else camera.name)
        self.__emit_parameter("resolution", "{0} {1}".format(width, height))
        self.__emit_custom_prop(scene, "color_space", "srgb")
        self.__close_element("frame")

    def __get_frame_resolution(self, render):
        scale = render.resolution_percentage / 100.0
        width = int(render.resolution_x * scale)
        height = int(render.resolution_y * scale)
        return width, height

    def __get_frame_aspect_ratio(self, render):
        width, height = self.__get_frame_resolution(render)
        xratio = width * render.pixel_aspect_x
        yratio = height * render.pixel_aspect_y
        return xratio / yratio

    #----------------------------------------------------------------------------------------------
    # Configurations.
    #----------------------------------------------------------------------------------------------

    def __emit_configurations(self):
        self.__open_element("configurations")
        self.__emit_interactive_configuration_element()
        self.__emit_final_configuration_element()
        self.__close_element("configurations")

    def __emit_interactive_configuration_element(self):
        self.__open_element('configuration name="interactive" base="base_interactive"')
        self.__emit_common_configuration_parameters()
        self.__close_element("configuration")

    def __emit_final_configuration_element(self):
        self.__open_element('configuration name="final" base="base_final"')
        self.__emit_common_configuration_parameters()
        self.__open_element('parameters name="generic_tile_renderer"')
        self.__emit_parameter("min_samples", self.sample_count)
        self.__emit_parameter("max_samples", self.sample_count)
        self.__close_element("parameters")
        self.__close_element("configuration")

    def __emit_common_configuration_parameters(self):
        self.__emit_parameter("lighting_engine", self.lighting_engine)
        self.__open_element('parameters name="{0}"'.format(self.lighting_engine))
        self.__emit_parameter("enable_ibl", "true" if self.enable_ibl else "false")
        self.__emit_parameter("enable_caustics", "true" if self.enable_caustics else "false")
        self.__close_element('parameters')

    #----------------------------------------------------------------------------------------------
    # Common elements.
    #----------------------------------------------------------------------------------------------

    def __emit_color_element(self, name, color_space, values, alpha, multiplier):
        self.__open_element('color name="{0}"'.format(name))
        self.__emit_parameter("color_space", color_space)
        self.__emit_parameter("multiplier", multiplier)
        self.__emit_line("<values>{0}</values>".format(" ".join(map(str, values))))
        if alpha:
            self.__emit_line("<alpha>{0}</alpha>".format(" ".join(map(str, alpha))))
        self.__close_element("color")

    #
    # A note on color spaces:
    #
    # Internally, Blender stores colors as linear RGB values, and the numeric color values
    # we get from color pickers are linear RGB values, although the color swatches and color
    # pickers show gamma corrected colors. This explains why we pretty much exclusively use
    # __emit_solid_linear_rgb_color_element() instead of __emit_solid_srgb_color_element().
    #

    def __emit_solid_linear_rgb_color_element(self, name, values, multiplier):
        self.__emit_color_element(name, "linear_rgb", values, None, multiplier)

    def __emit_solid_srgb_color_element(self, name, values, multiplier):
        self.__emit_color_element(name, "srgb", values, None, multiplier)

    def __emit_transform_element(self, m):
        #
        # We have the following conventions:
        #
        #   Both Blender and appleseed use right-hand coordinate systems.
        #   Both Blender and appleseed use column-major matrices.
        #   Both Blender and appleseed use pre-multiplication.
        #   In Blender, given a matrix m, m[i][j] is the element at the i'th row, j'th column.
        #
        # The only difference between the coordinate systems of Blender and appleseed is the up vector:
        # in Blender, up is Z+; in appleseed, up is Y+. We can go from Blender's coordinate system to
        # appleseed's one by rotating by +90 degrees around the X axis. That means that Blender objects
        # must be rotated by -90 degrees around X before being exported to appleseed.
        #

        self.__open_element("transform")
        self.__open_element("matrix")
        self.__emit_line("{0} {1} {2} {3}".format( m[0][0],  m[0][1],  m[0][2],  m[0][3]))
        self.__emit_line("{0} {1} {2} {3}".format( m[2][0],  m[2][1],  m[2][2],  m[2][3]))
        self.__emit_line("{0} {1} {2} {3}".format(-m[1][0], -m[1][1], -m[1][2], -m[1][3]))
        self.__emit_line("{0} {1} {2} {3}".format( m[3][0],  m[3][1],  m[3][2],  m[3][3]))
        self.__close_element("matrix")
        self.__close_element("transform")

    def __emit_custom_prop(self, object, prop_name, default_value):
        value = self.__get_custom_prop(object, prop_name, default_value)
        self.__emit_parameter(prop_name, value)

    def __get_custom_prop(self, object, prop_name, default_value):
        if prop_name in object:
            return object[prop_name]
        else:
            return default_value

    def __emit_parameter(self, name, value):
        self.__emit_line("<parameter name=\"" + name + "\" value=\"" + str(value) + "\" />")

    #----------------------------------------------------------------------------------------------
    # Utilities.
    #----------------------------------------------------------------------------------------------

    def __open_element(self, name):
        self.__emit_line("<" + name + ">")
        self.__indent()

    def __close_element(self, name):
        self.__unindent()
        self.__emit_line("</" + name + ">")

    def __emit_line(self, line):
        self.__emit_indent()
        self._output_file.write(line + "\n")

    def __indent(self):
        self._indent += 1

    def __unindent(self):
        assert self._indent > 0
        self._indent -= 1

    def __emit_indent(self):
        IndentSize = 4
        self._output_file.write(" " * self._indent * IndentSize)

    def __error(self, message):
        self.__print_message("error", message)
        self.report({ 'ERROR' }, message)

    def __warning(self, message):
        self.__print_message("warning", message)
        self.report({ 'WARNING' }, message)

    def __info(self, message):
        if len(message) > 0:
            self.__print_message("info", message)
        else: print("")
        self.report({ 'INFO' }, message)

    def __progress(self, message):
        self.__print_message("progress", message)

    def __print_message(self, severity, message):
        max_length = 8  # length of the longest severity string
        padding_count = max_length - len(severity)
        padding = " " * padding_count
        print("[{0}] {1}{2} : {3}".format(script_name, severity, padding, message))


#--------------------------------------------------------------------------------------------------
# Hook into Blender.
#--------------------------------------------------------------------------------------------------

def menu_func(self, context):
    default_path = os.path.splitext(bpy.data.filepath)[0] + ".appleseed"
    self.layout.operator(AppleseedExportOperator.bl_idname, text="appleseed (.appleseed)").filepath = default_path

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_export.append(menu_func)

def unregister():
    bpy.types.INFO_MT_file_export.remove(menu_func)
    bpy.utils.unregister_module(__name__)


#--------------------------------------------------------------------------------------------------
# Entry point.
#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    register()
