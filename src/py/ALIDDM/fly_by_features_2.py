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
    output_dir = os.path.join(args.out, "best_nets")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(dataset(args.dir))
    df_train, df_val = train_test_split(df, test_size=args.test_size)
    # print(df_train)
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
    # print(args.run_folder)
    writer = SummaryWriter(os.path.join(args.run_folder,"runs"))
    
    best_deplacment = 9999
    test_interval = args.test_interval

    agents = [Agent(phong_renderer, feat_net, device) for i in range(args.num_agents)]

    parameters = list(feat_net.parameters())

    for a in agents:
        parameters += a.get_parameters()

    optimizer = torch.optim.Adam(parameters, learning_rate)
    # optimizer = torch.optim.Adam([list(agents[aid].attention.parameters()) + list(agents[aid].delta_move.parameters())], learning_rate)

    
    for epoch in range(args.num_epoch):
        agents_ids = np.arange(args.num_agents)
        np.random.shuffle(agents_ids)
                    
        print('---------- epoch :', epoch,'----------')
        print('-------- TRAINING --------')
        Training(agents, agents_ids, args.num_step, train_dataloader, loss_function, optimizer, epoch_loss, device)

        if (epoch) % test_interval == 0:
            print('-------- VALIDATION --------')
            print('---------- epoch :', epoch,'----------')
            Validation(epoch,agents,agents_ids,test_dataloader,args.num_step,loss_function,best_deplacment,output_dir,device)
    
        if (epoch + 1) % args.num_epoch == 0:
            print('-------- ACCURACY --------')
            Accuracy(agents,test_dataloader,agents_ids,args.min_variance,loss_function,writer,device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir', type=str, help='dataset directory, if provided, it will be concatenated to the surf,landmarkrs file names', default='')
    # input_param.add_argument('--csv', type=str, help='csv with columns surf,landmarks,landmarks_number the landmarks column is a json filename with fiducials', required=True)
    # input_param.add_argument('--data_pred', type=str, help='dataset prediction', required=True)
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    input_param.add_argument('--test_size',type=int, help='proportion of dat for validation', default=0.1)
    input_param.add_argument('--batch_size',type=int, help='batch size', default=5)
    input_param.add_argument('--test_interval',type=int, help='when we do a evaluation of the model', default=5)
    input_param.add_argument('--run_folder',type=str, help='where you save tour run', default='/home/jonas/Desktop/Baptiste_Baquero/data_O')
    # input_param.add_argument('--run_folder',type=str, help='where you save tour run', default='/Users/luciacev-admin/Desktop/data_O')
    input_param.add_argument('--min_variance',type=float, help='minimum of variance', default=0.1)
    input_param.add_argument('--num_agents',type=int, help=' umber of agents = number of maximum of landmarks in dataset', default=2)
    input_param.add_argument('--num_step',type=int, help='number of step before to rich the landmark position',default=5)
    input_param.add_argument('--num_epoch',type=int,help="numero epoch", required=True)

    output_param = parser.add_argument_group('output files')
    output_param.add_argument('--out', type=str, help='place where model is saved', default='/home/jonas/Desktop/Baptiste_Baquero/data_O')
    # output_param.add_argument('--out', type=str, help='place where model is saved', default='/Users/luciacev-admin/Desktop/data_O')

    args = parser.parse_args()
    main(args)