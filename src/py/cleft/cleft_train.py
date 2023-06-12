import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from cleft_dataset import CleftDataset, CleftDataModule, TrainTransform, EvalTransform
from cleft_nets import CleftClassification
from cleft_logger import CleftImageLogger

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


    cleft_data = CleftDataModule(df_train, df_val, df_test,
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

    image_logger = CleftImageLogger()



    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
    class_weights = unique_class_weights

    model = CleftClassification(class_weights=class_weights, out_classes=len(class_weights), **vars(args))

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
    trainer.fit(model, datamodule=cleft_data, ckpt_path=args.model)

    trainer.test(datamodule=teeth_data)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Teeth challenge Training')
    parser.add_argument('--csv_train', help='CSV with column surf', type=str, required=True)    
    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, required=True)
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)        
    parser.add_argument('--surf_column', help='Surface column name', type=str, default="ID")
    parser.add_argument('--class_column', help='Class column name', type=str, default="Classification_0-2")        
    parser.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    parser.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    parser.add_argument('--image_size', help='Image resolution size', type=float, default=256)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/CMF/data/jprieto/juan_flyby/DCBIA/cleft/2.Retouched_Segs")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=6)        
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    parser.add_argument('--profiler', help='Use a profiler', type=str, default=None)

    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="cleft_classification")


    args = parser.parse_args()

    main(args)

