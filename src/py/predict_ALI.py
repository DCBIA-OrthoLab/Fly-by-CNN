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

from monai.config import print_config
import SimpleITK as sitk
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

def SavePrediction(data, outpath):
    print("Saving prediction to : ", outpath)
    img = data.numpy()
    output = sitk.GetImageFromArray(img)
    
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)

def Loader(data):
    list_img = []
    for img in data:
        # print(img["model"])
        # input_img = sitk.ReadImage(img["model"])
        # img_model = sitk.GetArrayFromImage(input_img)
        input_img = sitk.ReadImage(img)
        img_model = sitk.GetArrayFromImage(input_img) #rpz les 16 images 2D de 256x256
        
        for image in img_model:
            # print(torch.from_numpy(image).size())
            # dic={"model":torch.from_numpy(image).permute(2,0,1),"landmarks":torch.from_numpy(l_img[i]).permute(2,0,1)}
            new_image = torch.from_numpy(image).permute(2,0,1)
            list_img.append(new_image)
            # print(list_img)
    
    return list_img


def main(args):

    datalist = []
    
    if args.dir:
        normpath = os.path.normpath("/".join([args.dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd"]]:
                # img_obj = {}
                # img_obj["model"] = img_fn
                # img_obj["out"] = args.out
                # datalist.append(img_obj)
                datalist.append(img_fn)
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    # print(datalist)
    list_data = Loader(datalist) # return a list
    print("num data loaded",len(list_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=2,
        in_channels=7,
        out_channels=7,
        channels=(64, 128, 256, 512),
        strides=(2,2,2)
    ).to(device)

    print("loading model :", args.load_model)
    net.load_state_dict(torch.load(args.load_model,map_location=device))
    net.eval()
    print("Loading data from :", args.dir)
    
 
    
    with torch.no_grad():
        for img in datalist:
            print("Reading:", datalist)
            input_img = sitk.ReadImage(img)
            img_model = sitk.GetArrayFromImage(input_img) #rpz les 16 images 2D de 256x256 array
            output_filename = os.path.join(os.path.basename(img).split('.')[0] + '_predicted.nrrd')
            output_path = os.path.join(args.out,output_filename)
            
            for image in img_model:
                new_image = torch.from_numpy(image).permute(2,0,1) # convertion in tensor (7,258,258)
                img_output = net(new_image)
                # print(torch.from_numpy(img_output).size())
                output = torch.cat(img_output,0)
                # print(val_outputs.size())
                # out_img = torch.argmax(val_outputs, dim=1).detach().cpu()
                # out_img = out_img.type(torch.int16)
                                        
            SavePrediction(output, output_path)
            

    print("Done : " + str(len(datalist)) + " landmark predicted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)

    # input_group.add_argument('--load_model', type=str, help='Path of the model', default=None, required=True)

    input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)

    args = parser.parse_args()
    
    main(args)
