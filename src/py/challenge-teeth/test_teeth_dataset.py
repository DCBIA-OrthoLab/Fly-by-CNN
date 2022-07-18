
import os
import pandas as pd

import numpy as np

from teeth_dataset import TeethDataset, TeethDataModule, RandomRemoveTeethTransform, UnitSurfTransform
from teeth_nets import MonaiUNet

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

import SimpleITK as sitk

# def pad_verts_faces(batch):

#     s = [f.shape for v, f, rid, ridf, fpid0, cn in batch]
#     print(s)

#     verts = [v for v, f, rid, ridf, fpid0, cn in batch]
#     faces = [f for v, f, rid, ridf, fpid0, cn in batch]
#     verts_data = [vd for v, f, vd, vdf, fpid0, cn in batch]
#     verts_data_faces = [vdf for v, f, vd, vdf, fpid0, cn in batch]
#     faces_pid0s = [fpid0 for v, f, rid, ridf, fpid0, cn in batch]
#     color_normals = [cn for v, f, rid, ridf, fpid0, cn in batch]        
    
#     verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
#     faces = pad_sequence(faces, batch_first=True, padding_value=-1)        
#     verts_data = pad_sequence(verts_data, batch_first=True, padding_value=0.0)
#     verts_data_faces = pad_sequence(verts_data_faces, batch_first=True, padding_value=0.0)        
#     faces_pid0s = pad_sequence(faces_pid0s, batch_first=True, padding_value=-1)        
#     color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)

#     return verts, faces, verts_data, verts_data_faces, faces_pid0s, color_normals

def pad_verts_faces(batch):

    s = [f.shape for v, f, rid, ridf, cn in batch]

    verts = [v for v, f, rid, ridf, cn in batch]
    faces = [f for v, f, rid, ridf, cn in batch]
    verts_data = [vd for v, f, vd, vdf, cn in batch]
    verts_data_faces = [vdf for v, f, vd, vdf, cn in batch]        
    color_normals = [cn for v, f, rid, ridf, cn in batch]        
    
    verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
    faces = pad_sequence(faces, batch_first=True, padding_value=-1)        
    verts_data = pad_sequence(verts_data, batch_first=True, padding_value=0.0)
    verts_data_faces = torch.cat(verts_data_faces)
    color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)

    return verts, faces, verts_data, verts_data_faces, color_normals

mount_point = "/work/leclercq/data/challenge_teeth_vtk"

df_train = pd.read_csv(os.path.join(mount_point, "train.csv"))

train_ds = TeethDataset(df_train, mount_point=mount_point, surf_property = "UniversalID", transform=UnitSurfTransform(random_rotation=True)) #RandomRemoveTeethTransform(surf_property="UniversalID", random_rotation=True)

train_loader = DataLoader(train_ds, batch_size=6, num_workers=1, prefetch_factor=1, shuffle=True, collate_fn=pad_verts_faces)

model = MonaiUNet()
model.cuda()

labels = np.array([])
for batch in tqdm(train_loader, total=len(df_train)):
    V, F, Y, YF, CN = batch 

    x, X, PF = model((V, F, CN))
    
    y = torch.take(YF.cuda(), PF)*(PF >= 0) # YF=input, pix_to_face=index. shape of y_p=shape of pix_to_face

    y = y.permute(0, 1, 3, 4, 2).cpu().numpy()
    
    X = X.permute(0, 1, 3, 4, 2).cpu().numpy()    

    PF = PF.permute(0, 1, 3, 4, 2).cpu().numpy()

    sitk.WriteImage(sitk.GetImageFromArray(X[2, :, :, :, 0:3], isVector=True), "temp.nrrd")
    sitk.WriteImage(sitk.GetImageFromArray(y[2], isVector=True), "temp_label.nrrd")
    sitk.WriteImage(sitk.GetImageFromArray(PF[2], isVector=True), "temp_pf.nrrd")
    
    quit()


#   labels = np.concatenate([labels, Y.numpy().reshape(-1)])

# class_weight_vect = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

# print(np.unique(labels), class_weight_vect)

# np.save(open(os.path.join(mount_point, "train_weights.npy"), 'wb'), class_weight_vect)