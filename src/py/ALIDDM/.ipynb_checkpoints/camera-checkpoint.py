import torch 
import argparse
from post_process import ReadFile
from utils import *
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import matplotlib.pyplot as plt



def main(args):

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")


    surf = ReadSurf(args.dir_data)
    verts,faces = PolyDataToTensors(surf)
    print(type(faces))
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    meshes = Meshes(verts=[verts], faces=[faces],textures=textures)
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    R, T = look_at_view_transform(2.7, 0, 180) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    print("apres la camera")
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    print("rasterization")
    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    print("light")
    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
    
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    print("renderer")
    images = renderer(meshes)
    # print(images)
    # print(torch.Size(images))
    print("image")
    plt.figure(figsize=(50, 50))
    print('salut')
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='separAte all the teeth from a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir_data', type=str, help='Input directory with 3D images', required=True)

    # output_params = parser.add_argument_group('Output parameters')
    # output_params.add_argument('--out', type=str, help='Output directory', )
    
    args = parser.parse_args()
    main(args)