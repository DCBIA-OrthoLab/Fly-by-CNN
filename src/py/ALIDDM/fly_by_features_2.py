#!/usr/bin/env python
# coding: utf-8

from shutil import move
from itk.support.extras import image
from pytorch3d import renderer
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
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    df = pd.read_csv(dataset(args.dir))
    dt = pd.read_csv(dataset(args.test))

    df_train, df_val = train_test_split(df, train_size=args.train_size)
    print(df_train.shape)
    print(df_val.shape)
    print(dt.shape)
    # print(df_train)
    # df_prediction = dataset(args.data_pred)

    train_data = FlyByDataset(df_train,device, dataset_dir=args.dir, rotate=True)
    val_data = FlyByDataset(df_val,device , dataset_dir=args.dir, rotate=True)
    test_data = FlyByDataset(dt,device,dataset_dir=args.dir, rotate=False)


    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
    validation_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=pad_verts_faces)
    
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=pad_verts_faces)

    learning_rate = 1e-4
    feat_net = FeaturesNet().to(device)
    # new_move_net = TimeDistributed(move_net).to(device)
    loss_function = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    early_stopping = EarlyStopping(patience=10, verbose=True, path=args.out)

    epoch_loss = 0
    best_score = 9999
    # print(args.run_folder)
    # writer = SummaryWriter(os.path.join(args.run_folder,"runs"))

    agents = [Agent(renderer=phong_renderer, features_net=feat_net,run_folder=args.run_folder, aid=i, device=device) for i in range(args.num_agents)]

    parameters = list(feat_net.parameters())

    for a in agents:
        parameters += a.get_parameters()

    optimizer = torch.optim.Adam(parameters, learning_rate)
    # optimizer = torch.optim.Adam([list(agents[aid].attention.parameters()) + list(agents[aid].delta_move.parameters())], learning_rate)

    
    for epoch in range(args.num_epoch):
        agents_ids = np.arange(args.num_agents)
        np.random.shuffle(agents_ids)

        print('-------- TRAINING --------')          
        print('---------- epoch :', epoch,'----------')
        Training(epoch, agents, agents_ids, args.num_step, train_dataloader, loss_function, optimizer, device)

        if (epoch) % args.test_interval == 0:
            print('-------- VALIDATION --------')
            print('---------- epoch :', epoch,'----------')
            Validation(epoch,agents,agents_ids,validation_dataloader,args.num_step,loss_function,output_dir,early_stopping,device)
            if early_stopping.early_stop == True :
                print('-------- ACCURACY --------')
                Accuracy(agents,test_dataloader,agents_ids,args.min_variance,loss_function,device)
                break
        
        if (epoch + 1) % args.num_epoch == 0:
            print('-------- ACCURACY --------')
            Accuracy(agents,test_dataloader,agents_ids,args.min_variance,loss_function,device)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir', type=str, help='dataset directory, if provided, it will be concatenated to the surf,landmarkrs file names', default='')
    # input_param.add_argument('--csv', type=str, help='csv with columns surf,landmarks,landmarks_number the landmarks column is a json filename with fiducials', required=True)
    # input_param.add_argument('--data_pred', type=str, help='dataset prediction', required=True)
    input_param.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_param.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_param.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    input_param.add_argument('--train_size',type=int, help='proportion of dat for training', default=0.9)
    input_param.add_argument('--test',type=str, help='all the datas for testing', default='' )

    input_param.add_argument('--batch_size',type=int, help='batch size', default=10)
    input_param.add_argument('--test_interval',type=int, help='when we do a evaluation of the model', default=1)
    input_param.add_argument('--run_folder',type=str, help='where you save tour run', default='./runs')
    input_param.add_argument('--min_variance',type=float, help='minimum of variance', default=0.1)
    input_param.add_argument('--num_agents',type=int, help=' number of agents = number of maximum of landmarks in dataset', default=2)
    input_param.add_argument('--num_step',type=int, help='number of step before to rich the landmark position',default=10)
    input_param.add_argument('--num_epoch',type=int,help="numero epoch", required=True)

    output_param = parser.add_argument_group('output files')
    output_param.add_argument('--out', type=str, help='place where model is saved', default='./training/')
    # output_param.add_argument('--out', type=str, help='place where model is saved', default='/Users/luciacev-admin/Desktop/data_O')

    args = parser.parse_args()
    main(args)