import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch import nn
from torch.utils.data import DataLoader

from cleft_dataset import CleftDataset, CleftDataModule, TrainTransform, EvalTransform
from cleft_nets import CleftClassification
from cleft_logger import CleftImageLogger

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


    test_ds = CleftDataset(df_test, transform=EvalTransform(), **vars(args))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    model = CleftClassification(**vars(args)).load_from_checkpoint(args.model)
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
        
        probs.append(x)
        predictions.append(torch.argmax(x, dim=1, keepdim=True))


    probs = torch.cat(probs).detach().cpu().numpy()
    predictions = torch.cat(predictions).cpu().numpy().squeeze()

    out_dir = os.path.join(args.out, os.path.basename(args.model))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pickle.dump(probs, open(os.path.join(out_dir, fname.replace(ext, "_probs.pickle")), 'wb'))

    df_test['pred'] = predictions
    if ext == ".csv":
        df_test.to_csv(os.path.join(out_dir, fname.replace(ext, "_prediction.csv")), index=False)
    else:
        df_test.to_parquet(os.path.join(out_dir, fname.replace(ext, "_prediction.parquet")), index=False)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Cleft predict')    
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)   
    parser.add_argument('--model', help='Model for prediction', type=str, required=True)
    parser.add_argument('--out_classes', help='Number of classes', type=int, default=3)

    parser.add_argument('--surf_column', help='Surface column name', type=str, default="ID")
    parser.add_argument('--class_column', help='Class column name', type=str, default="Classification_0-2")        
    parser.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    parser.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    parser.add_argument('--image_size', help='Image resolution size', type=float, default=256)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/CMF/data/jprieto/juan_flyby/DCBIA/cleft/2.Retouched_Segs")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)

    args = parser.parse_args()

    main(args)

