import logging
import os
import sys
import tempfile
from glob import glob
import math

from sklearn.model_selection import train_test_split
import pandas as pd
import SimpleITK as sitk 
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    Compose,
    RandRotated,
    ScaleIntensityd,
    ToTensord,
    EnsureType,
    Activations, 
    AsDiscrete
)
from monai.visualize import plot_2d_or_3d_image

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}. Best validation loss: {self.val_loss_min:.6f} <-- {val_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
       
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class DatasetGenerator(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.loc[idx]
        img = row["img"]
        seg = row["seg"]

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img)).reshape((512, 512, 3)).astype(float)
        seg_np = sitk.GetArrayFromImage(sitk.ReadImage(seg)).reshape((512, 512)).astype(float)

        return {"img": img_np, "seg": seg_np}
    

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
    print(
        f"Rank {rank + 1}/{world_size} process initialized.\n"
    )
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    df = pd.read_csv("/work/jprieto/data/remote/EGower/jprieto/eyes_cropped_resampled_512_seg_train.csv")

    train_df, valid_df = train_test_split(df, test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            AsChannelFirstd(keys=["img"]),
            AddChanneld(keys=["seg"]),
            RandRotated(keys=["img", "seg"], prob=0.5, range_x=math.pi/2.0, range_y=math.pi/2.0, mode=["bilinear", "nearest"]),
            ScaleIntensityd(keys=["img"]),
            ToTensord(keys=["img", "seg"])
        ]
    )
    val_transforms = Compose(
        [
            AsChannelFirstd(keys=["img"]),
            AddChanneld(keys=["seg"]),
            ScaleIntensityd(keys=["img"]),
            ToTensord(keys=["img", "seg"])
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=DatasetGenerator(train_df), transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=8,
        num_workers=8,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    
    # create a validation data loader
    val_ds = monai.data.Dataset(data=DatasetGenerator(valid_df), transform=val_transforms)

    val_sampler = DistributedSampler(val_ds, shuffle=False, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_ds, sampler=val_sampler, batch_size=2, num_workers=8, collate_fn=list_data_collate)


    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer    

    # model = monai.networks.nets.UNETR(in_channels=3, out_channels=4, img_size=(512,512), pos_embed='conv', norm_name='instance', spatial_dims=2).to(device)    
    model = monai.networks.nets.UNet(spatial_dims=2, in_channels=3, out_channels=4, channels=(16, 32, 64, 128, 256, 512, 1024), strides=(2, 2, 2, 2, 2, 2), num_res_units=4).to(device)
    model = DDP(model, device_ids=[device])

    loss_function = monai.losses.DiceLoss(softmax=True, to_onehot_y=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    # writer = SummaryWriter()

    num_epochs = 100
    early_stop = EarlyStopping(patience=10, verbose=True,
        path='train/torch_unet_train_01252022/model.pt')

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")

        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:

            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if rank == 0:
                epoch_len = len(train_loader)
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        dist.all_reduce(epoch_loss)
        
        if rank == 0:
            epoch_loss = epoch_loss.item()/(step*world_size)
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            val_loss = 0.0
            step = 0
            for val_data in val_loader:
                step += 1
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                roi_size = (512, 512)
                # sw_batch_size = 4
                # val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                # val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                val_outputs = model(val_images)
                val_loss += loss_function(outputs, labels)
                val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
                dice_metric(y_pred=val_outputs, y=val_labels)
            # aggregate the final mean dice result
            metric = dice_metric.aggregate()
            # reset the status for next validation round
            dice_metric.reset()            
            dist.all_reduce(metric)

            dist.all_reduce(val_loss)
            val_loss = val_loss.cpu().item()/(step*world_size)

            if rank == 0:
                early_stop(val_loss, model.module)            
                if early_stop.early_stop:
                    early_stop_indicator = torch.tensor([1.0]).to(device)
                else:
                    early_stop_indicator = torch.tensor([0.0]).cuda()
            else:
                early_stop_indicator = torch.tensor([0.0]).to(device)

            dist.all_reduce(early_stop_indicator)

            if early_stop_indicator.cpu().item() == 1.0:
                print("Early stopping")            
                break

    if rank == 0:
        print(f"train completed")

    cleanup()



WORLD_SIZE = torch.cuda.device_count()
if __name__ == "__main__":
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9999'

    mp.spawn(
        main, args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE, join=True
    )
