from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh import textures
import torch 
import numpy as np
import random
import argparse
from post_process import ReadFile
from utils import *
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import (PerspectiveCameras,look_at_rotation)
from pytorch3d.renderer import (
    rasterize_meshes,
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

from preprocess import *
import matplotlib.pyplot as plt
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from utils_cam import *

def main(args):
    
    distance = 4

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    surf_LM = ReadSurf(args.dir_landmarks)
    surf_merged = ReadSurf(args.dir_data)

    unit_surf_merged, mean_arr, scale_factor = ScaleSurf(surf_merged)
    unit_surf_LM = ScaleSurf(surf_LM, mean_arr=mean_arr, scale_factor=scale_factor)

    # normal_list = get_normal(surf_merged,landmarks_file)
    # if args.random_rotation:
    #     unit_surf_LM, rotationAngle, rotationVector = RandomRotation(unit_surf_LM)
    #     unit_surf_merged = RotateSurf(unit_surf_merged, rotationAngle, rotationVector)

    # surf = fbf.ComputeNormals(surf)

    verts_teeth,faces_teeth = PolyDataToTensors(unit_surf_merged)
    verts_rgb = torch.ones_like(verts_teeth)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    meshe_teeth = Meshes(
        verts=[verts_teeth], 
        faces=[faces_teeth],
        textures=textures
    )

    # fig = plot_scene({"subplot1": {"teeth_mesh": meshe_teeth}})
    # fig.show()
    # verts,faces = PolyDataToTensors(unit_surf_LM)
    # meshe_landmarks = Meshes(verts=[verts], faces=[faces])
    # fig2 = plot_scene({"subplot1": {"teeth_mesh": meshe_landmarks}})
    # fig2.show()
    # print([0.0] * 3)
    # shapedatapoints = unit_surf_LM.GetPoints()
    # bounds = shapedatapoints.GetBounds()
    # print(bounds)
    # print(torch.max(verts,0))
    # print(torch.min(verts,0))
    
    vector_cam = np.array([distance,0,0])


########################################################################
# initialization camera
########################################################################
    theta_x,theta_y,theta_z = set_random_rot()
    position_initial_cam = rotation(vector_cam,theta_x,theta_y,theta_z)
    # print(position_initial_cam)
    cam_pos = ToTensor(dtype=torch.float32,device=device)(position_initial_cam)
    camera = FoVPerspectiveCameras(device=device)
    
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0, 
        faces_per_pixel=1 )
    
    lights = PointLights(device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
            ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights
            )
        )

    R = look_at_rotation(cam_pos,device=device)  # (1, 3, 3)
    elevation, azimuth, _ = set_random_rot()
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device,degrees=False)
    images = renderer(meshes_world=meshe_teeth,R=R,T=T)
    images= images.cpu().numpy()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(images.squeeze())
    plt.axis("on")
    
    elevation, azimuth, _ = set_random_rot()
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device,degrees=False)
    images = renderer(meshes_world=meshe_teeth,R=R,T=T)
    images= images.cpu().numpy()
   
    plt.subplot(2, 2, 2)
    plt.imshow(images.squeeze())
    plt.axis("on")
    
    elevation, azimuth, _ = set_random_rot()
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device,degrees=False)
    images = renderer(meshes_world=meshe_teeth,R=R,T=T)
    images= images.cpu().numpy()

    plt.subplot(2, 2, 3)
    plt.imshow(images.squeeze())
    plt.axis("on")

    plt.show()

    # print(vector_cam.shape)
    # print(vector_cam)

    # cam_pos = torch.from_numpy(position_initial_cam).type(torch.float32)
    # print(torch.from_numpy(position_initial_cam))
    # print(position_initial_cam)
    # print(position_initial_cam[0][0],position_initial_cam[1][0],position_initial_cam[2][0])
    # print(np.linalg.norm(position_initial_cam))
    # print(position_initial_cam[0])
    # print(position_initial_cam[1])
    # print(position_initial_cam[2])

    
    # print(camera.get_camera_center())
    # print(camera.get_world_to_view_transform())






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='separAte all the teeth from a vtk file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir_data', type=str, help='Input crown directory', required=True)
    input_param.add_argument('--dir_landmarks',type=str, help='Input landmarks directory')
    
    
    data_augment_parser = parser.add_argument_group('Data augment parameters')
    data_augment_parser.add_argument('--random_rotation', type=bool, help='activate or not a random rotation', default=True)

    # output_params = parser.add_argument_group('Output parameters')
    # output_params.add_argument('--out', type=str, help='Output directory', )

    args = parser.parse_args()
    main(args)