import bpy
import os

# Specify the directory where the images are located
img_dir = '/work/jprieto/data/DCBIA/condyle_4_class0_frontr_out/'

# Get a list of all files in directory
img_list = os.listdir(img_dir)

# Filter out all non-.jpg files
img_list = [img for img in img_list if img.endswith('.png')] # Adjust file extension if needed

# Sort the images by name
img_list.sort()

# Get the length of the image list
len_img_list = len(img_list)

# Specify the output path
output_path = '/work/jprieto/data/DCBIA/condyle_4_class0_frontr_out.mp4'

# Set render settings
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.fps = 5 # or the frame rate you want
bpy.context.scene.render.filepath = output_path

# Create new image texture
bpy.ops.image.open(filepath=os.path.join(img_dir, img_list[0]), directory=img_dir, files=[{"name":img_list[0]}], relative_path=True, show_multiview=False)

# Create new material
mat = bpy.data.materials.new('ImgMaterial')

# Enable 'Use Nodes'
bpy.context.object.active_material.use_nodes = True

# Add texture node to material
texture_node = mat.node_tree.nodes.new('ShaderNodeTexImage')

# Load image sequence
texture_node.image_user.frame_duration = len_img_list
texture_node.image_user.frame_start = 1
texture_node.image_user.frame_offset = 0
texture_node.image.source = 'SEQUENCE'

# Render animation
bpy.ops.render.render(animation=True, scene=bpy.context.scene.name)