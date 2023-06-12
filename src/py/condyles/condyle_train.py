import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from condyle_dataset import CondyleDataset, CondyleDataModule, TrainTransform, EvalTransform
import condyle_nets
# from condyle_nets import CondyleClassification
from condyle_logger import CondyleImageLogger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
        
    mount_point = args.mount_point    

    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_valid))


    condyle_data = CondyleDataModule(df_train, df_val, df_test,
                                mount_point = mount_point,
                                batch_size = args.batch_size,
                                num_workers = args.num_workers,
                                surf_column = args.surf_column, class_column=args.class_column,
                                train_transform = TrainTransform(),
                                valid_transform = EvalTransform(),
                                test_transform = EvalTransform())

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    image_logger = CondyleImageLogger()



    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
    class_weights = unique_class_weights

    # model = CondyleClassification(class_weights=class_weights, out_classes=len(class_weights), **vars(args))
    
    NN = getattr(condyle_nets, args.nn)
    model = NN(class_weights=class_weights, out_classes=len(class_weights), **vars(args))

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=0,
        profiler=args.profiler
    )
    trainer.fit(model, datamodule=condyle_data, ckpt_path=args.model)

    os.symlink(checkpoint_callback.best_model_path, os.path.join(args.out, "best_model.ckpt"))    
    # trainer.test(datamodule=teeth_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Condyle training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=20)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=6)
    hparams_group.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    hparams_group.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    hparams_group.add_argument('--image_size', help='Image resolution size', type=float, default=256)
    hparams_group.add_argument('--norm_zbuf', help='Normalize z_buf', type=int, default=0)

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="CondyleClassification")
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    input_group.add_argument('--class_column', help='Class column name', type=str, default="class")            

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--profiler', help='Use a profiler', type=str, default=None)
    log_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    


    args = parser.parse_args()

    main(args)

