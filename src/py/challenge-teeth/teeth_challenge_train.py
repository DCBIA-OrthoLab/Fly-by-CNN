import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from teeth_dataset import TeethDataset, TeethDataModule, RandomRemoveTeethTransform, UnitSurfTransform
from teeth_nets import MonaiUNet
from teeth_logger import TeethNetImageLogger

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

    # class_weights = np.load(os.path.join(mount_point, 'train_weights.npy'))
    class_weights = None

    model = MonaiUNet(args, out_channels = 34, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples)

    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))


    teeth_data = TeethDataModule(df_train, df_val, 
                                mount_point = mount_point,
                                batch_size = args.batch_size,
                                num_workers = args.num_workers,
                                surf_column = 'surf', surf_property="UniversalID",
                                train_transform = RandomRemoveTeethTransform(surf_property="UniversalID", random_rotation=True),
                                valid_transform = UnitSurfTransform(),
                                test_transform = UnitSurfTransform())

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    image_logger = TeethNetImageLogger()

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
        num_sanity_val_steps=0,
        profiler=args.profiler
    )
    trainer.fit(model, datamodule=teeth_data, ckpt_path=args.model)

    trainer.test(ckpt_path="best")


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, required=True)    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, required=True)
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)        
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/work/leclercq/data/challenge_teeth_vtk")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=6)    
    parser.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)    
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="monai")


    args = parser.parse_args()

    main(args)

