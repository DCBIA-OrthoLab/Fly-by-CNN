import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from teeth_dataset import TeethDataset, RandomRemoveTeethTransform, UnitSurfTransform
from teeth_nets import MonaiUNet

import utils

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import nrrd

def main(args):
    
        
    mount_point = args.mount_point

    # class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))
    class_weights = None
    out_channels = 34

    model = MonaiUNet(args, out_channels = out_channels, class_weights=class_weights, image_size=320, subdivision_level=2)

    model.model.module.load_state_dict(torch.load(args.model))

    df = pd.read_csv(os.path.join(mount_point, args.csv))

    ds = TeethDataset(df, mount_point = args.mount_point, transform=UnitSurfTransform(), surf_column="surf")

    dataloader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    

    device = torch.device('cuda')
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=2)

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            V, F, CN = batch

            V = V.cuda(non_blocking=True)
            F = F.cuda(non_blocking=True)
            CN = CN.cuda(non_blocking=True).to(torch.float32)

            x, X, PF = model((V, F, CN))
            x = softmax(x*(PF>=0))

            P_faces = torch.zeros(out_channels, F.shape[1]).to(device)
            V_labels_prediction = torch.zeros(V.shape[1]).to(device).to(torch.int64)

            PF = PF.squeeze()
            x = x.squeeze()

            for pf, pred in zip(PF, x):
                P_faces[:, pf] += pred

            P_faces = torch.argmax(P_faces, dim=0)

            faces_pid0 = F[0,:,0]
            V_labels_prediction[faces_pid0] = P_faces

            surf = ds.getSurf(idx)

            V_labels_prediction = numpy_to_vtk(V_labels_prediction.cpu().numpy())
            V_labels_prediction.SetName(args.array_name)
            surf.GetPointData().AddArray(V_labels_prediction)

            output_fn = os.path.join(args.out, df["surf"][idx])

            output_dir = os.path.dirname(output_fn)

            if(not os.path.exists(output_dir)):
                os.makedirs(output_dir)

            utils.Write(surf , output_fn, print_out=False)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge prediction')
    parser.add_argument('--csv', help='CSV with column surf', type=str, required=True)    
    parser.add_argument('--model', help='Model to continue training', type=str, default="/work/leclercq/data/07-21-22_val-loss0.169.pth")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--array_name',type=str, help = 'Predicted ID array name for output vtk', default="PredictedID")


    args = parser.parse_args()

    main(args)

