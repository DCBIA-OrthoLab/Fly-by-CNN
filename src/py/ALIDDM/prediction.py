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
    
    attention_lst = []
    move_net_lst = []
    normpath = os.path.normpath("/".join([args.load_models, '**', '']))
    for model in sorted(glob.iglob(normpath, recursive=True)):
        if True in ['_feature_' in model]:
            feature_net_path = model
        if True in ['_attention_' in model]:
            attention_lst.append(model)
        if True in ['_delta_move_' in model]:
            move_net_lst.append(model)
    
    # print(feature_net_path)
    # print(attention_lst)
    # print(move_net_lst)
    print("Loading data from :", args.dir)
    df = pd.read_csv(dataset(args.dir))
    df_train, df_rem = train_test_split(df, train_size=args.train_size)
    df_val, data = train_test_split(df_rem, test_size=args.test_size )
    
    data = FlyByDatasetPrediction(df,device, dataset_dir=args.dir)
    dataloader = DataLoader(data, batch_size=args.batch_size, collate_fn=pad_verts_faces_prediction)
   
    feat_net = FeaturesNet().to(device)
    agents = [Agent(renderer=phong_renderer, features_net=feat_net, aid=i, device=device) for i in range(args.num_agents)]
    agents_ids = np.arange(args.num_agents)
    print(agents_ids)
    # writer = SummaryWriter(os.path.join(args.run_folder,"runs"))
    
    print("loading feature net ... :", feature_net_path )
    feat_net = torch.load(feature_net_path,map_location=device)
    out_path = os.path.join(args.jsonfolder,'Lower_jaw.json')

    for idx_agent,model in enumerate(attention_lst):
        print("loading attention net ... :", model)
        agents[idx_agent].attention = torch.load(model,map_location=device)

    for idx_agent,model in enumerate(move_net_lst):
        print("loading move net ... :", model)
        agents[idx_agent].delta_move = torch.load(model,map_location=device)
        
        

    print('-------- PREDICTION --------')
    groupe_data = Prediction(agents,dataloader,agents_ids,args.min_variance)
    lm_lst = GenControlePoint(groupe_data)
    WriteJson(lm_lst,out_path)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)
    input_group.add_argument('--min_variance',type=float, help='minimum of variance', default=0.3)
    # input_group.add_argument('--run_folder',type=str, help='where you save tour run', default='/Users/luciacev-admin/Desktop/data_O')
    input_group.add_argument('--train_size',type=int, help='proportion of dat for validation', default=0.7)
    input_group.add_argument('--test_size',type=int, help='proportion of dat for validation', default=0.6)

    # input_group.add_argument('--load_models', type=str, help='Path of the model', default='/Users/luciacev-admin/Desktop/data_O/best_move_net')
    input_group.add_argument('--load_models', type=str, help='Path of the model', default='/home/jonas/Desktop/Baptiste_Baquero/data_O/best_nets')
    input_group.add_argument('--num_agents',type=int, help=' umber of agents = number of maximum of landmarks in dataset', default=2)
    input_group.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_group.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_group.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    input_group.add_argument('--batch_size',type=int, help='batch size', default=2)
    input_group.add_argument('--jsonfolder',type=str, help='path where you save your jsonfile after prediction', required=True)

    
    args = parser.parse_args()
    
    main(args)
