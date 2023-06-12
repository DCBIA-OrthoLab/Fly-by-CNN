
# Dental Model Segmentation Challenge

- Contributors: Mathieu Leclerq, Baptiste Baquero, Marcela Gurgel, Lucia Cevidanes, Martin Styner, Juan Prieto

This folder contains all the code necesary to run training for the dental model segmentation challenge. 

## Prepare the data for training

```
python fly-by-cnn/src/py/challenge-teeth/obj_to_vtk.py -h
usage: obj_to_vtk.py [-h] --csv CSV [--out OUT]

Teeth challenge convert OBJ files to VTK. It adds the label information to each VTK file

optional arguments:
  -h, --help  show this help message and exit
  --csv CSV   CSV with columns surf,label,split
  --out OUT   Output directory

```

An example input CSV file [challenge_teeth_all.csv](./challenge_teeth_all.csv). The CSV file must contain columns surf,label,split

```
python fly-by-cnn/src/py/challenge-teeth/obj_to_vtk.py --csv challenge_teeth_all.csv --out teeth-grand_challenge_vtk/
```

This command will create the corresponding splits and also the files "test.csv",  "train.csv"  and "val.csv". These are used to start the training.

## Install all required dependencies

```
conda env create -f flyby.yml
```
## Start training

```
CUDA_VISIBLE_DEVICES=0,1 python fly-by-cnn/src/py/challenge-teeth/teeth_challenge_train.py --csv_train teeth-grand_challenge_vtk/train.csv --csv_valid teeth-grand_challenge_vtk/val.csv --csv_test teeth-grand_challenge_vtk/test.csv --out training_output/ --tb_dir tensorboard/ --tb_name flyby --num_workers 16 --batch_size 60
```
