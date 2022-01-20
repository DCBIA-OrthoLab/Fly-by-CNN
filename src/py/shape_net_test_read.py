import shape_net_dataset as snd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils
import nrrd
import os
import numpy as np
import sys

data_dir = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1"
csv_split = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1/all.csv"

snd_train = snd.ShapeNetDataset(data_dir, csv_split=csv_split, split="train")


for idx, model in snd_train.df.sample(frac=1).iterrows():

    model_path = os.path.join(snd_train.shapenet_dir, model["synsetId"], model["modelId"], snd_train.model_dir)
    model_path_vtk = model_path.replace(".obj", ".vtk")

    try:
        # if not os.path.exists(model_path_vtk):
        #     print("Reading:", model_path)
        #     surf = utils.ReadSurf(model_path)
        #     print("Writting:", model_path_vtk)
        #     utils.WriteSurf(surf, model_path_vtk)
        print("Reading:", model_path_vtk)
        surf = utils.ReadSurf(model_path_vtk)
    except:
        print("CANNOT READ:", model_path, file=sys.stderr)