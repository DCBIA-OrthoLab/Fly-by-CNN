from model import UNet
# from train_utils import *
import monai
import argparse
import os
import glob
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from monai.metrics import ROCAUCMetric
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import decollate_batch, partition_dataset_classes
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SaveImaged,
    EnsureTyped,
    EnsureType,
    Activations,
    
)

from monai.config import print_config



from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch


def main(args):

    model_lst = []
    landmarks_lst = []
    datalist = []

#############   Get Data   #################

    if args.dir:
        normpath = os.path.normpath("/".join([args.dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd"]]:
                if True in [name in img_fn for name in ["_RCSeg_"]]:
                    model_lst.append(img_fn)
            
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd"]]:
                if True in [name in img_fn for name in ["_landmarks_"]]:
                    landmarks_lst.append(img_fn)
        # print(model_lst)
        # print(landmarks_lst)

    if len(model_lst) != len(landmarks_lst):
        print("ERROR : Not the same number of models and landmarks file")
        return
    
    for file_id in range(0,len(model_lst)):
        data = {"model" : model_lst[file_id], "landmarks" : landmarks_lst[file_id]}
        datalist.append(data)
    # print(datalist)
    # define dataset
    TrainingSet , TestSet = train_test_split(datalist, test_size=0.10, train_size=0.90, random_state=len(datalist) )


# #############      Load Data        ##################


    # define transforme for model and landmarks
    train_transform = Compose(
        [
            LoadImaged(keys=["model", "landmarks"]),
            AddChanneld(keys=["model", "landmarks"],channel_dim=-1),
            ScaleIntensityd(keys="model"),
            EnsureTyped(keys=["model", "landmarks"]),
        ]
    )
    validation_transform = Compose(       
        [
            LoadImaged(keys=["model", "landmarks"]),
            AddChanneld(keys=["model", "landmarks"],channel_dim=-1),
            ScaleIntensityd(keys="model"),
            EnsureTyped(keys=["model", "landemarks"]),
        ]
    )
    

    # define dataset, dataloader
    Train_dataset = CacheDataset(        
        data=TrainingSet,
        transform=train_transform,
        # cache_num=24,
        # cache_rate=1.0,
        # num_workers=4,
    )

    Train_loader = DataLoader(
        Train_dataset, 
        batch_size=20, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    # how to choose the parameters
    
    check_data = monai.utils.misc.first(Train_loader) 
    print(check_data["model"].shape, check_data["landmarks"].shape)
    
    # create a validation data loader
    validation_dataset = CacheDataset(
        data=TestSet, 
        transform=validation_transform, 
        # cache_num=6, 
        # cache_rate=1.0, 
    )
    val_loader = DataLoader(
        validation_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    to_onehot = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=num_class)])
# # ############# Network, Losses, Optimizer #############
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# # #############  Training ##################
    
    epoch_num = args.epoch
    # val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    auc_metric = ROCAUCMetric() 
    
    for epoch in range(epoch_num):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{epoch_num}")

        epoch_loss = 0
        step = 1
        steps_per_epoch = len(Train_dataset) // Train_loader.batch_size

        # put the network in train mode; this tells the network and its modules to
        # enable training elements such as normalisation and dropout, where applicable
        model.train()

        for batch_data in Train_loader:
            step += 1
            # move the data to the GPU
            inputs, landmarks = batch_data["model"].to(device), batch_data["landmarks"].to(device)
            # prepare the gradients for this step's back propagation
            optimizer.zero_grad()
            # run the network forwards
            outputs = model(inputs)
            # run the loss function on the outputs
            loss = loss_function(outputs, landmarks)
            # compute the gradients
            loss.backward()
             # tell the optimizer to update the weights according to the gradients
            # and its internal optimisation strategy
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(Train_dataset) // Train_loader.batch_size
            print(f"{step}/{steps_per_epoch+1}, train_loss: {loss.item():.4f}")
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            step+=1
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            # switch off training features of the network for this pass
            model.eval()
            # 'with torch.no_grad()' switches off gradient calculation for the scope of its context
            with torch.no_grad():
                # create lists to which we will concatenate the the validation resulte
                preds = list()
                labels = list()
                for val_data in val_loader:
                    val_images, val_labels = val_data["model"].to(device), val_data["landmarks"].to(device)
                    # run the network
                    val_pred = model(val_images)
                    preds.append(val_pred)
                    labels.append(val_labels)
                    
                    # concatenate the predicted labels with each other and the actual labels with each other
                    y_pred = torch.cat(preds)
                    y = torch.cat(labels)

                 
                    # we are using the area under the receiver operating characteristic (ROC) curve to determine
                    # whether this epoch has improved the best performance of the network so far, in which case
                    # we save the network in this state
                    y_onehot = [to_onehot(i) for i in decollate_batch(y)]        
                    y_pred_act = [act(i) for i in decollate_batch(y_pred)]
                    
                    auc_metric(y_pred_act, y_onehot)
                    auc_value = auc_metric.aggregate()
                    auc_metric.reset()
                    metric_values.append(auc_value)
                    
                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    
                    if auc_value > best_metric:
                        best_metric = auc_value
                        best_metric_epoch = epoch + 1
                        torch.save(net.state_dict(), os.path.join(args.dir, "best_metric_model.pth"))
                        print("saved new best metric network")
                        
                    print(
                        f"current epoch: {epoch + 1} current AUC: {auc_value:.4f} /"
                        f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f} /"
                        f" at epoch: {best_metric_epoch}"
                    )

                    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='training a unet model to predict landmarks on model 3D of teeth', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_param = parser.add_argument_group('input files')
    input_param.add_argument('--dir', type=str, help='path directory', required=True)

    input_param.add_argument('--epoch', type=int, help='number of epochs ', required=True)
    
    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output directory', required=True)
   
    args = parser.parse_args()
    
    main(args)