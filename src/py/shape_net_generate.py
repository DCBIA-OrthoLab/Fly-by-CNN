import shape_net_dataset as snd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils
import nrrd
import os
import numpy as np

data_dir = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1"
csv_split = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1/all.csv"

snd_train = snd.ShapeNetDataset(data_dir, csv_split=csv_split, split="val", use_vtk=True)

output_dataset_dir = "/work/jprieto/data/ShapeNet/ShapeNetCore.v1_nrrd"



for idx, model in snd_train.df.sample(frac=1).iterrows():

    # for epoch in range(30):
    for epoch in range(1):

        output_path = os.path.join(output_dataset_dir, model["synsetId"], model["modelId"], str(epoch))

        output_path_normals = output_path + "_norm.nrrd"
        output_path_zbuffer = output_path + "_z.nrrd"

        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(output_path_normals) or not os.path.exists(output_path_zbuffer):

            print("Reading:", os.path.join(model["synsetId"], model["modelId"], snd_train.model_dir))
            img_np, img_np_z, synset_class = snd_train[idx]
            
            print("Writting:", output_path_normals)
            header = {'kinds': ['domain', 'domain', 'domain', 'vector']}
            nrrd.write(output_path_normals, img_np, header)

            print("Writting:", output_path_zbuffer)
            header = {'kinds': ['domain', 'domain', 'domain', 'scalar']}
            nrrd.write(output_path_zbuffer, img_np_z, header)