import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn
from torch.utils.data import DataLoader

from condyle_dataset import CondyleDataset, CondyleDataModule, TrainTransform, EvalTransform
import condyle_nets

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2

import pickle
from tqdm import tqdm

from monai.transforms import (    
    ScaleIntensityRange
)


import post_process as psp
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def RemoveIslands(vtkdata, labels, label, min_count=1000,ignore_neg1 = False):

    pid_visited = np.zeros(labels.GetNumberOfTuples())
    for pid in range(labels.GetNumberOfTuples()):
        if labels.GetTuple(pid)[0] == label and pid_visited[pid] == 0:
            connected_pids = ConnectedRegion(vtkdata, pid, labels, label, pid_visited)
            if connected_pids.shape[0] < min_count:
                neighbor_label = NeighborLabel(vtkdata, labels, label, connected_pids)
                if ignore_neg1 == True and neighbor_label != -1:
                    for cpid in connected_pids:
                        labels.SetTuple(int(cpid), (neighbor_label,))

def main(args):
    
    fname = os.path.basename(args.csv_test)    
    ext = os.path.splitext(fname)[1]

    if ext == ".csv":
        df_test = pd.read_csv(args.csv_test)
    else:
        df_test = pd.read_parquet(args.csv_test)


    test_ds = CondyleDataset(df_test, transform=EvalTransform(), **vars(args))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)


    NN = getattr(condyle_nets, args.nn)
    model = NN.load_from_checkpoint(args.model)
    model.ico_sphere(radius=args.radius, subdivision_level=args.subdivision_level)

    device = torch.device('cuda')
    model = model.to(device)

    model.eval()

    if hasattr(model.F.module, "features"):
        target_layers = [model.F.module.features[-1]]
    else:
        target_layers = [model.F.module.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(args.target_class)]

    scale_intensity = ScaleIntensityRange(0.0, 1.0, 0, 255)

    out_dir = os.path.join(args.out, os.path.basename(args.model), "grad_cam", str(args.target_class))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        V = V.cuda(non_blocking=True)
        F = F.cuda(non_blocking=True)
        CN = CN.cuda(non_blocking=True)

        X, PF = model.render(V, F, CN)        

        gcam_np = cam(input_tensor=X, targets=targets)

        GCAM = torch.tensor(gcam_np).to(device)

        P_faces = torch.zeros(1, F.shape[1]).to(device)
        V_gcam = -1*torch.ones(V.shape[1], dtype=torch.float32).to(device)

        for pf, gc in zip(PF.squeeze(), GCAM):
            P_faces[:, pf] = torch.maximum(P_faces[:, pf], gc)

        faces_pid0 = F[0,:,0].to(torch.int64)
        V_gcam[faces_pid0] = P_faces

        surf = test_ds.getSurf(idx)

        V_gcam = numpy_to_vtk(V_gcam.cpu().numpy())
        array_name = "grad_cam_target_class_{target_class}".format(target_class=args.target_class)
        V_gcam.SetName(array_name)
        surf.GetPointData().AddArray(V_gcam)

        psp.MedianFilter(surf, V_gcam)

        surf_path = df_test.loc[idx][args.surf_column]
        ext = os.path.splitext(surf_path)[1]

        if ext == '':
            ext = ".vtk"
            surf_path += ext

        out_surf_path = os.path.join(out_dir, surf_path)

        if not os.path.exists(os.path.dirname(out_surf_path)):
            os.makedirs(os.path.dirname(out_surf_path))

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(out_surf_path)
        writer.SetInputData(surf)
        writer.Write()


        X = (X*(PF>=0)).cpu().numpy()        
        vid_np = scale_intensity(X).permute(0,1,3,4,2).squeeze().cpu().numpy().squeeze().astype(np.uint8)        
        gcam_np = scale_intensity(gcam_np).squeeze().numpy().astype(np.uint8)

        
        out_vid_path = surf_path.replace(ext, '.mp4')
        out_vid_path = os.path.join(out_dir, out_vid_path)

        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_vid_path, fourcc, args.fps, (256, 256))

        for v, g in zip(vid_np, gcam_np):
            c = cv2.applyColorMap(g, cv2.COLORMAP_JET)            
            b = cv2.addWeighted(v[:,:,0:3], 0.5, c, 0.5, 0)
            out.write(b)

        out.release()



    out_dir = os.path.join(args.out, os.path.basename(args.model))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # pickle.dump(probs, open(os.path.join(out_dir, fname.replace(ext, "_probs.pickle")), 'wb'))

    # df_test['pred'] = predictions
    # if ext == ".csv":
    #     df_test.to_csv(os.path.join(out_dir, fname.replace(ext, "_prediction.csv")), index=False)
    # else:
    #     df_test.to_parquet(os.path.join(out_dir, fname.replace(ext, "_prediction.parquet")), index=False)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Cleft predict')    
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)   
    parser.add_argument('--model', help='Model for prediction', type=str, required=True)
    parser.add_argument('--nn', help='Type of neural network', type=str, default="CondyleClassification")
    parser.add_argument('--eigen_smooth', help='Smooth the output. It only works for the maximum response class', type=int, default=0)
    parser.add_argument('--out_classes', help='Number of classes', type=int, default=4)
    parser.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    parser.add_argument('--class_column', help='Class column name', type=str, default="class")        
    parser.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    parser.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    parser.add_argument('--image_size', help='Image resolution size', type=float, default=256)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/work/jprieto/data/DCBIA")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--target_class', help='Target class', type=int, default=0)
    parser.add_argument('--fps', help='Frames per second', type=int, default=24)

    args = parser.parse_args()

    main(args)

