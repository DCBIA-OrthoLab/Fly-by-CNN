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

import pickle
from tqdm import tqdm

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
    model.eval()

    cuda = torch.device('cuda')
    model = model.to(cuda)

    probs = []
    predictions = []
    softmax = nn.Softmax(dim=1)

    for idx, (V, F, CN, L) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        V = V.cuda(non_blocking=True)
        F = F.cuda(non_blocking=True)
        CN = CN.cuda(non_blocking=True)

        X, PF = model.render(V, F, CN)
        x = model(X)
        
        x = softmax(x)
        
        probs.append(x.detach())
        predictions.append(torch.argmax(x, dim=1, keepdim=True).detach())


    probs = torch.cat(probs).cpu().numpy()
    predictions = torch.cat(predictions).cpu().numpy().squeeze()

    out_dir = os.path.join(args.out, os.path.basename(args.model))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    probs_fn = os.path.join(out_dir, fname.replace(ext, "_probs.pickle"))    
    pickle.dump(probs, open(probs_fn, 'wb'))

    df_test['pred'] = predictions
    if ext == ".csv":
        df_test.to_csv(os.path.join(out_dir, fname.replace(ext, "_prediction.csv")), index=False)
    else:
        df_test.to_parquet(os.path.join(out_dir, fname.replace(ext, "_prediction.parquet")), index=False)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Condyle predict')    

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)   
    input_group.add_argument('--model', help='Model for prediction', type=str, required=True)    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="CondyleClassification")    
    input_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    input_group.add_argument('--class_column', help='Class column name', type=str, default="class")        

    load_group = parser.add_argument_group('Loading')
    load_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/work/jprieto/data/DCBIA")
    load_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    load_group.add_argument('--batch_size', help='Batch size', type=int, default=1)    

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    hparams_group.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    # hparams_group.add_argument('--image_size', help='Image resolution size', type=float, default=256)    
    # hparams_group.add_argument('--out_classes', help='Number of classes', type=int, default=4)
    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")        

    args = parser.parse_args()

    main(args)

