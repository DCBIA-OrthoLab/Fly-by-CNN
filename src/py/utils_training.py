import SimpleITK as sitk
from monai.data.utils import DataFrame
import torch

# ----- MONAI ------

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
# from monai.transforms import (
#     AddChanneld,
#     Compose,
#     ScaleIntensityd,
#     ToTensor,
#     transform
    
# )

import numpy as np
import json
import os
data_type = torch.float32



def Loader(data):
    list_img = []
    for img in data:
        input_img = sitk.ReadImage(img["model"])
        img_model = sitk.GetArrayFromImage(input_img)
        # print(np.shape(img_model))
        landmarks_img = sitk.ReadImage(img["landmarks"])
        l_img = sitk.GetArrayFromImage(landmarks_img)
        
        for i,image in enumerate(img_model):
            # print(torch.from_numpy(image).size())
            dic={"model":torch.from_numpy(image).permute(2,0,1),"landmarks":torch.from_numpy(l_img[i]).permute(2,0,1)}
        
            list_img.append(dic)
        
    return list_img

