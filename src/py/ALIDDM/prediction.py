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
    
    net_lst = []
    normpath = os.path.normpath("/".join([args.load_models, '**', '']))
    for model in sorted(glob.iglob(normpath, recursive=True)):
        if True in ['.pth' in model]:
            net_lst.append(model)
    print(net_lst)

    df = pd.read_csv(dataset(args.dir))
    df_train, df_val = train_test_split(df, test_size=args.test_size)
    train_data = FlyByDataset(df_train,device, dataset_dir=args.dir)
    val_data = FlyByDataset(df_val,device,  dataset_dir=args.dir)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
    test_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_verts_faces)
   
    # df = pd.read_csv(dataset(args.dir))
    # data = FlyByDataset(df,device, dataset_dir=args.dir)
    loss_function = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    feat_net = FeaturesNet().to(device)
    agents = [Agent(phong_renderer, feat_net, device) for i in range(args.num_agents)]
    agents_ids = np.arange(args.num_agents)
    print(agents_ids)
    writer = SummaryWriter(os.path.join(args.run_folder,"runs"))
    agents_lst = []
   
    for idx_agent,model in enumerate(net_lst):
        print("loading model :", model)
        agents[idx_agent].load_state_dict(torch.load(model,map_location=device))
        agents_lst.append(agents[idx_agent])
        
    print("Loading data from :", args.dir)

    print('-------- ACCURACY --------')
    Accuracy(agents_lst,test_dataloader,agents_ids,args.min_variance,loss_function,writer,device)

        
        # list_distance = ({ 'obj' : [], 'distance' : [] })
        
   

        # with torch.no_grad():
        #     for batch, (V, F, CN, LP) in enumerate(data):

        #         textures = TexturesVertex(verts_features=CN)
        #         meshes = Meshes(
        #             verts=V,   
        #             faces=F, 
        #             textures=textures
        #         )
        #         for aid in agents_ids: #aid == idlandmark_id
        #             print('---------- agents id :', aid,'----------')

        #             agents[aid].eval() 
                    
        #             pos_center = agents[aid].search(meshes,min_variance) #[batchsize,3]
                    
        #             lm_pos = torch.empty((0)).to(device)
        #             for lst in LP:
        #                 lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)  #[batchsize,3]
                    
        #             loss = loss_function(pos_center, lm_pos)
                    
        #             list_distance['obj'].append(aid)
        #             list_distance['distance'].append(loss)
                    
        #             writer.add_scalar('distance',loss)
            
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)
    input_group.add_argument('--min_variance',type=float, help='minimum of variance', default=0.1)
    input_group.add_argument('--run_folder',type=str, help='where you save tour run', default='/Users/luciacev-admin/Desktop/data_O')
    input_group.add_argument('--load_models', type=str, help='Path of the model', default='/Users/luciacev-admin/Desktop/data_O/best_move_net')
    input_group.add_argument('--num_agents',type=int, help=' umber of agents = number of maximum of landmarks in dataset', default=2)
    input_group.add_argument('--image_size',type=int, help='size of the picture', default=224)
    input_group.add_argument('--blur_radius',type=int, help='blur raius', default=0)
    input_group.add_argument('--faces_per_pixel',type=int, help='faces per pixels', default=1)
    input_group.add_argument('--test_size',type=int, help='proportion of dat for validation', default=0.1)
    input_group.add_argument('--batch_size',type=int, help='batch size', default=10)
    
    args = parser.parse_args()
    
    main(args)
