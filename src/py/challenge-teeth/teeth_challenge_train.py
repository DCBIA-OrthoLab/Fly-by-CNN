import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from teeth_dataset import TeethDataset, TeethDataModule, RandomRemoveTeethTransform, UnitSurfTransform
from teeth_nets import MonaiUNet

from pl_bolts.models.self_supervised import Moco_v2

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

# from azureml.core.run import Run
# run = Run.get_context()

def main(args):


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
        
    mount_point = args.mount_point

    class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))

    model = MonaiUNet(args, out_channels = 34, class_weights=class_weights, image_size=320)

    df_train = pd.read_csv(os.path.join(mount_point, "train_merged.csv"))
    df_val = pd.read_csv(os.path.join(mount_point, "val_merged.csv"))
    df_test = pd.read_csv(os.path.join(mount_point, "test.csv"))


    teeth_data = TeethDataModule(df_train, df_val, df_test, 
                                mount_point = mount_point,
                                batch_size = args.batch_size,
                                num_workers = args.num_workers,
                                surf_column = 'surf', surf_property="UniversalID",
                                train_transform = RandomRemoveTeethTransform(surf_property="UniversalID", random_rotation=True),
                                valid_transform = UnitSurfTransform())

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    trainer.fit(model, datamodule=teeth_data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/work/leclercq/data/challenge_teeth_vtk")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=30)    
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")


    args = parser.parse_args()

    main(args)

