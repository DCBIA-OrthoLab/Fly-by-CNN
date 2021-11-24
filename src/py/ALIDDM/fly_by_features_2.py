#!/usr/bin/env python
# coding: utf-8

from shutil import move
from itk.support.extras import image
import torch
from torch._C import default_generator
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,
)
import sys
sys.path.insert(0,'..')
from sklearn.model_selection import train_test_split
from utils_cam import *
from utils_class import *
import argparse
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import random

def main(args):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    df = pd.read_csv(args.csv)
    df_train, df_val = train_test_split(df, test_size=args.test_size)
    # df_prediction = dataset(args.data_pred)

    train_data = FlyByDataset(df_train,device, dataset_dir=args.dir)
    val_data = FlyByDataset(df_val,device,  dataset_dir=args.dir)
    # data_prediction = FlyByDataset(df_prediction,2,device)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
    test_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
    # pred_dataloader = DataLoader(data_prediction,batch_size=args.batch_size,shuffle=True, collate_fn=pad_verts_faces)
    # print(pred_dataloader)

    learning_rate = 1e-5
    feat_net = FeaturesNet().to(device)
    # new_move_net = TimeDistributed(move_net).to(device)
    loss_function = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    
    epoch_loss = 0
    writer = SummaryWriter(os.path.join(args.run_folder,"runs"))
    
    best_deplacment = 99999
    best_deplacment_epoch = 0
    test_interval = args.test_interval
    list_distance = []

    agents = [Agent(phong_renderer, feat_net, device, batch_size = args.batch_size) for i in range(42)]

    parameters = list(feat_net.parameters())

    for a in agents:
        parameters += list(a.parameters())

    optimizer = torch.optim.Adam(parameters, learning_rate)
    # optimizer = torch.optim.Adam([list(agents[aid].attention.parameters()) + list(agents[aid].delta_move.parameters())], learning_rate)

    
    for epoch in range(args.num_epoch):
        print('---------- epoch :', epoch,'----------')
        agents_ids = np.arange(42)
        np.random.shuffle(agents_ids)

        for batch, (V, F, CN, LP) in enumerate(train_dataloader):
            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            
            # list_pictures = agent.shot(meshes)
            # agent.affichage(list_pictures)
            img_batch = torch.empty((0)).to(device)

            for aid in agents_ids: #aid == idlandmark_id
                print('---------- agents id :', aid,'----------')

                NSteps = 10
                step_loss = 0
            
                agents[aid].trainable(True)

                for i in range(NSteps):
                    print('---------- step :', i,'----------')

                    optimizer.zero_grad()   # prepare the gradients for this step's back propagation

                    x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]
                    
                    x += agents[aid].sphere_centers
                    # print('coord sphere center :', agent.sphere_center)
                    loss = loss_function(x, LP)

                    loss.backward()   # backward propagation
                    optimizer.step()   # tell the optimizer to update the weights according to the gradients and its internal optimisation strategy
                    
                    step_loss += loss.item()
                    agents[aid].sphere_centers = x.detach().clone()
                
                step_loss /= NSteps
                agents[aid].trainable(False)

                print("Step loss:", step_loss)
                epoch_loss += step_loss
            # agent.affichage(list_pictures)
    

    # affichage(train_dataloader,phong_renderer)
            
    # for epoch in range(args.num_epoch):
    #     print('-------- TRAINING --------')

    #     training( epoch, move_net, train_dataloader, phong_renderer, loss_function, optimizer, epoch_loss, writer, device)

    #     if (epoch +1 ) % test_interval == 0:
    #         print('-------- VALIDATION --------')
    #         print(epoch +1)
    #         validation(epoch,move_net,test_dataloader,phong_renderer,loss_function,list_distance,best_deplacment,best_deplacment_epoch,args.out,device)
    
    # print('-------- ACCURACY --------')
    # Accuracy(move_net,test_dataloader,phong_renderer,args.min_variance,loss_function,writer,device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir', type=str, help='dataset directory, if provided, it will be concatenated to the surf,landmarkrs file names', default='')
    input_param.add_argument('--csv', type=str, help='csv with columns surf,landmarks,landmarks_number the landmarks column is a json filename with fiducials', required=True)
    # input_param.add_argument('--data_pred', type=str, help='dataset prediction', required=True)
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=24)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    input_param.add_argument('--test_size',type=int, help='proportion of dat for validation', default=0.5)
    input_param.add_argument('--batch_size',type=int, help='batch size', default=5)
    input_param.add_argument('--test_interval',type=int, help='when we do a evaluation of the model', default=5)
    input_param.add_argument('--run_folder',type=str, help='where you save tour run', default='/Users/luciacev-admin/Desktop/data_O')
    input_param.add_argument('--min_variance',type=float, help='minimum of variance', default=0.1)

    parser.add_argument('--num_epoch',type=int,help="numero epoch",required=True)

    output_param = parser.add_argument_group('output files')
    output_param.add_argument('--out', type=str, help='place where model is saved', default='/Users/luciacev-admin/Desktop/data_O')

    args = parser.parse_args()
    main(args)