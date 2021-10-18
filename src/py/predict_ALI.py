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
from utils_training import *

from monai.config import print_config
import SimpleITK as sitk
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
# def CreatePredictTransform(data):
#             data_type = torch.float32
#             pre_transforms = Compose([AddChanneld(),ScaleIntensityd(minv = 0.0, maxv = 1.0, factor = None)])
#             input_img = sitk.ReadImage(data) 
#             img = sitk.GetArrayFromImage(input_img)
#             pre_img = torch.from_numpy(pre_transforms(img))
#             pre_img = pre_img.type(data_type)
#             return pre_img,input_img

def SavePrediction(data,input_img, outpath):
    print("Saving prediction to : ", outpath)
    img = data.numpy()[0][:]
    output = sitk.GetImageFromArray(img)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)

def main(args):

    datalist = []
    
    if args.dir:
        normpath = os.path.normpath("/".join([args.dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
                img_obj = {}
                img_obj["model"] = img_fn
                img_obj["out"] = args.out
                datalist.append(img_obj)
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    
    list_data = Loader(datalist)
    
    filename = os.path.splitext(os.path.basename(model_file))[0]
    ouput_filename = os.path.join(args.dir_ft, filename + ".nrrd")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=7,
        out_channels=7,
        channels=(64, 128, 256, 512),
        strides=(2,2,2)
    ).to(device)

    print("loading model :", args.load_model)
    model.load_state_dict(torch.load(args.load_model,map_location=device))
    model.eval()
    print("Loading data from", args.dir)

    with torch.no_grad():
        for data in list_data:
            print("Reading:", data['model'])
            val_outputs = model(data['model'])
            # print(val_outputs.size())
            # out_img = torch.argmax(val_outputs, dim=1).detach().cpu()
            # out_img = out_img.type(torch.int16)
            
            baseName = os.path.basename(data["model"])
            modelname= baseName.split(".")[0]
            pred_name = ""
            for i,element in enumerate(modelname):
                if i == 0:
                    pred_name += element.replace("scan","Pred")
                else:
                    pred_name += "." + element
                        
            input_dir = os.path.dirname(data["image"])
            
            SavePrediction(out_img ,input_image , os.path.join(input_dir,pred_name))
            

    print("Done : " + str(len(datalist)) + " landmark predicted")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    input_group = parser.add_argument_group('directory')
    input_group.add_argument('--dir', type=str, help='Input directory with the scans',default=None, required=True)

    input_group.add_argument('--load_model', type=str, help='Path of the model', default=None, required=True)

    input_group.add_argument('--out', type=str, help='Output directory with the landmarks',default=None)

    args = parser.parse_args()
    
    main(args)
