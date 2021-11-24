from monai.networks.nets import UNet
import argparse
import os
import glob
from sklearn.model_selection import train_test_split
import torch
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.metrics import ROCAUCMetric
from monai.data import decollate_batch, partition_dataset_classes
import numpy as np
from utils_cam import *
from monai.config import print_config
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = dataset(args.dir)
    print(df)
    data = FlyByDataset(df,2,device)
    # dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
    move_net = MoveNet().to(device)

    print("loading model :", args.load_model)
    move_net.load_state_dict(torch.load(args.load_model,map_location=device))
    move_net.eval()
    print("Loading data from :", args.dir)
    
    cameras = FoVPerspectiveCameras(device=device) # Initialize a perspective camera.

    raster_settings = RasterizationSettings(        
        image_size=args.image_size, 
        blur_radius=args.blur_radius, 
        faces_per_pixel=args.faces_per_pixel, 
    )

    lights = PointLights(device=device) # light in front of the object. 

    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )

    phong_renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    
    writer = SummaryWriter(os.path.join(args.run_folder,"runs"))
    
    with torch.no_grad():

        for batch, (V, F, Y, F0, CN, IP) in enumerate(data):
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            camera_net = CameraNet(meshes, phong_renderer)
            landmark_coord = camera_net.search(move_net,args.min_variance) 
            images = camera_net.shot().to(device) 
            # distance = loss_function(camera_net.camera_position, IP)

            writer.add_images('image',images)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)
    input_group.add_argument('--min_variance',type=float, help='minimum of variance', default=0.5)
    input_group.add_argument('--run_folder',type=str, help='where you save tour run', default='/Users/luciacev-admin/Desktop/data_O')
    input_group.add_argument('--load_model', type=str, help='Path of the model', default='/Users/luciacev-admin/Desktop/best_move_net.pth')
    
    args = parser.parse_args()
    
    main(args)
