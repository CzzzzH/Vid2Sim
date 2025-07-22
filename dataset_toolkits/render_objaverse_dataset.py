import sys
sys.path.append(".")

import numpy as np
import argparse
import bpy
import os 
import json
import gc 
import time

from mathutils import Vector 
from utils.seeding import seed_everything 
 
def look_at(obj, target, objaverse=False):
    direction = obj.location - target
    rot_quat = direction.to_track_quat('Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def clear_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_camera(source, target, fov=None):
    
    # Set up the camera
    if fov is not None:
        camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        camera.data.angle = np.radians(fov)
        bpy.context.collection.objects.link(camera)
        bpy.context.scene.camera = camera
    else:
        camera = bpy.data.objects.get("Camera")
        
    camera.location = source
    look_at(camera, target)
    return camera

def setup_floor():
    
    floor_mesh = bpy.data.meshes.new("FloorMesh")
    floor = bpy.data.objects.new("Floor", floor_mesh)
    
    bpy.context.collection.objects.link(floor)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = floor
    floor.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.primitive_plane_add(size=10)  # Adjust the size as needed
    bpy.ops.object.mode_set(mode='OBJECT')
    
    floor_material = bpy.data.materials.new(name="FloorMaterial")
    floor_material.use_nodes = True
    
    nodes = floor_material.node_tree.nodes
    links = floor_material.node_tree.links
    nodes.clear()
    
    principled_bsdf = nodes.new(type="ShaderNodeBsdfDiffuse")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(principled_bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Albedo (Base Color)
    albedo_texture = nodes.new(type="ShaderNodeTexImage")
    albedo_texture.image = bpy.data.images.load("dataset_toolkits/assets/floor/albedo.jpg") 
    links.new(albedo_texture.outputs["Color"], principled_bsdf.inputs["Color"])

    floor.data.materials.append(floor_material)
    floor.location = (0, 0, -0.7)
    return floor

def setup_envmap():

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    nodes.clear()
    env_texture = nodes.new(type='ShaderNodeTexEnvironment') 
    env_texture.image = bpy.data.images.load("dataset_toolkits/assets/envmap/sunrise_sky_dome_4k.exr") 
    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.inputs['Strength'].default_value
    node_tree.links.new(background_node.inputs['Color'], env_texture.outputs['Color'])
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    node_tree.links.new(output_node.inputs['Surface'], background_node.outputs['Background'])

    mapping_node = nodes.get("Mapping") or nodes.new(type="ShaderNodeMapping")
    tex_coord_node = nodes.get("Texture Coordinate") or nodes.new(type="ShaderNodeTexCoord")
    node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    node_tree.links.new(mapping_node.outputs["Vector"], env_texture.inputs["Vector"])
    rotation_degrees = 270  # Adjust this angle as needed
    mapping_node.inputs["Rotation"].default_value[2] = np.radians(rotation_degrees)

def setup_rendering(render_size):
     
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.render.resolution_x = render_size
    bpy.context.scene.render.resolution_y = render_size
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.view_layer.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.scene.cycles.filter_width = 1.0
    bpy.context.scene.cycles.denoising_prefilter = 'FAST'
    bpy.context.view_layer.use_pass_diffuse_color = False
    bpy.context.scene.cycles.samples = 16

def hide_background(object_name):
     
    current_obj = get_object(object_name)
    current_obj.pass_index = 233
    bpy.context.view_layer.use_pass_object_index = True
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()
    
    render_layer_node = nodes.new(type="CompositorNodeRLayers")
    id_mask_node = nodes.new(type="CompositorNodeIDMask")
    id_mask_node.index = 233
    links.new(render_layer_node.outputs["IndexOB"], id_mask_node.inputs["ID value"])
    alpha_over_node = nodes.new(type="CompositorNodeAlphaOver")
    alpha_over_node.inputs[1].default_value = (1, 1, 1, 0)  # Set transparent background
    links.new(id_mask_node.outputs["Alpha"], alpha_over_node.inputs[0])
    links.new(render_layer_node.outputs["Image"], alpha_over_node.inputs[2])
    composite_node = nodes.new(type="CompositorNodeComposite")
    links.new(alpha_over_node.outputs["Image"], composite_node.inputs["Image"])
    
    # Set the floor object as a shadow catcher
    floor = get_object("Floor")
    bpy.context.view_layer.objects.active = floor
    floor.select_set(True)
    bpy.context.object.cycles.is_shadow_catcher = True  # Enable shadow catcher in Cycles

def get_object(name):
    for obj in bpy.data.objects:
        if name in obj.name:
            return obj
    return None

def clean():
    
    gc.collect()
    bpy.ops.outliner.orphans_purge()
    if bpy.data.images.get("Render Result"):
        bpy.data.images["Render Result"].user_clear()
        bpy.data.images.remove(bpy.data.images["Render Result"])
    
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
        
    for image in bpy.data.images:
        if image.users == 0:
            bpy.data.images.remove(image)

    for texture in bpy.data.textures:
        if texture.users == 0:
            bpy.data.textures.remove(texture)

    for node_group in bpy.data.node_groups:
        if node_group.users == 0:
            bpy.data.node_groups.remove(node_group)

def render_data(args):
 
    total_frame = 16

    with open(f"{args.dataset_dir}/objaverse_uid_list.json", "r") as f:
        obj_list = json.load(f)
    start_idx = max(args.start_idx, 0)
    end_idx = min(args.end_idx, len(obj_list))
    output_dir = f'{args.dataset_dir}/outputs'
    print(f'Render {end_idx - start_idx} objects from {start_idx} to {end_idx}')

    # Basic Settings
    render_size = 448
    camera_distance = 1.8
    fov = 45 
    
    source = Vector((0, -camera_distance, -0.2))
    target = Vector((0, 0, -0.2)) 
    
    clear_objects()
    setup_camera(fov=fov, source=source, target=target)
    setup_floor()
    setup_envmap()
    setup_rendering(render_size)

    for i, obj_path in enumerate(obj_list[start_idx:end_idx]): 

        obj_id = obj_path.split('/')[-1]
        seed_everything(obj_id)
        
        if not os.path.exists(f'{output_dir}/{obj_id}/meshes/{total_frame-1:03d}.glb'):
            print(f"Simulated mesh not found for {obj_id}, skip rendering")
            continue
            
        if os.path.exists(f'{output_dir}/{obj_id}/renderings/{total_frame-1:03d}.png'): 
            continue
            
        clean()
        os.makedirs(f'{output_dir}/{obj_id}/renderings', exist_ok=True)  
        start_time = time.time()

        bpy.ops.import_scene.gltf(filepath=f'{output_dir}/{obj_id}/meshes/{total_frame-1:03d}.glb')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.name != 'Floor':
                name = obj.name
        
        for k in range(total_frame):
            for obj in bpy.data.objects:
                if obj.name != 'Floor' and obj.name != 'Camera':
                    bpy.data.objects.remove(obj)
            bpy.ops.import_scene.gltf(filepath=f'{output_dir}/{obj_id}/meshes/{k:03d}.glb')
            bpy.context.scene.render.filepath = f'{output_dir}/{obj_id}/renderings/{k:03d}.png'
            bpy.context.scene.frame_set(k)    
            hide_background(name)
            bpy.ops.render.render(write_still=True)

        print(f'Success to render {obj_id} in {time.time() - start_time:.2f}s')             

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/objaverse")   
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50000)
    args = parser.parse_args()

    render_data(args) 
