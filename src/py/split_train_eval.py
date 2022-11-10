import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Splits data into train/eval', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--csv', type=str, help='CSV file', required=True)
parser.add_argument('--split', type=float, help='Split float [0-1]', default=0.0)
parser.add_argument('--group_by', type=str, help='Group the rows by column', default=None)
parser.add_argument('--folds', type=int, help='Number of folds to generate', default=0)
parser.add_argument('--csv_split', type=str, help='Split the data using the ids from this dataframe', default=None)

args = parser.parse_args()

fname = args.csv

if(os.path.splitext(fname)[1] == ".csv"):
    df = pd.read_csv(fname)
else:
    df = pd.read_parquet(fname)

split = args.split
if split == 0.0 and args.folds > 0:
    split = 1.0/args.folds

if args.csv_split:
    df_split = pd.read_csv(args.csv_split)


if args.group_by:

    if args.csv_split:
        group_ids = df_split[args.group_by]
    else:
        group_ids = df[args.group_by].unique()
        np.random.shuffle(group_ids)

    samples = int(len(group_ids)*split)

    if args.folds == 0:

        id_test = group_ids[0:samples]
        id_train = group_ids[samples:]


        df_train = df[df[args.group_by].isin(id_train)]
        df_test = df[df[args.group_by].isin(id_test)]

        if(os.path.splitext(fname)[1] == ".csv"):
            if split > 0:
                train_fn = fname.replace('.csv', '_train.csv')
                df_train.to_csv(train_fn, index=False)

                eval_fn = fname.replace('.csv', '_test.csv')
                df_test.to_csv(eval_fn, index=False)
            else:
                
                split_fn = fname.replace('.csv', '_split.csv')
                df_train.to_csv(split_fn, index=False)
        else:
            if split > 0:
                train_fn = fname.replace('.parquet', '_train.parquet')
                df_train.to_parquet(train_fn, index=False)

                eval_fn = fname.replace('.parquet', '_test.parquet')
                df_test.to_parquet(eval_fn, index=False)
            else:
                
                split_fn = fname.replace('.parquet', '_split.parquet')
                df_train.to_parquet(split_fn, index=False)


    else:

        start_f = 0
        end_f = samples
        for i in range(args.folds):

            id_test = group_ids[start_f:end_f]

            df_train = df[~df[args.group_by].isin(id_test)]
            df_test = df[df[args.group_by].isin(id_test)]

            if(os.path.splitext(fname)[1] == ".csv"):
                train_fn = fname.replace('.csv', 'fold' + str(i) + '_train.csv')
                df_train.to_csv(train_fn, index=False)

                eval_fn = fname.replace('.csv', 'fold' + str(i) + '_test.csv')
                df_test.to_csv(eval_fn, index=False)
            else:
                train_fn = fname.replace('.parquet', 'fold' + str(i) + '_train.parquet')
                df_train.to_parquet(train_fn, index=False)

                eval_fn = fname.replace('.parquet', 'fold' + str(i) + '_test.parquet')
                df_test.to_parquet(eval_fn, index=False)

            

            start_f += samples
            end_f += samples

else:
    group_ids = np.array(range(len(df.index)))

    samples = int(len(group_ids)*split)

    np.random.shuffle(group_ids)

    if args.folds == 0:

        id_test = group_ids[0:samples]
        id_train = group_ids[samples:]


        df_train = df.iloc[id_train]
        df_test = df.iloc[id_test]

        if(os.path.splitext(fname)[1] == ".csv"):
            if split > 0:

                train_fn = fname.replace('.csv', '_train.csv')
                df_train.to_csv(train_fn, index=False)

                eval_fn = fname.replace('.csv', '_test.csv')
                df_test.to_csv(eval_fn, index=False)
            else:

                split_fn = fname.replace('.csv', '_split.csv')
                df_train.to_csv(split_fn, index=False)
        else:
            if split > 0:

                train_fn = fname.replace('.parquet', '_train.parquet')
                df_train.to_parquet(train_fn, index=False)

                eval_fn = fname.replace('.parquet', '_test.parquet')
                df_test.to_parquet(eval_fn, index=False)
            else:

                split_fn = fname.replace('.parquet', '_split.parquet')
                df_train.to_parquet(split_fn, index=False)

    else:

        start_f = 0
        end_f = samples
        for i in range(args.folds):

            id_test = group_ids[start_f:end_f]

            df_train = df[~df.index.isin(id_test)]
            df_test = df.iloc[id_test]

            if(os.path.splitext(fname)[1] == ".csv"):
                train_fn = fname.replace('.csv', 'fold' + str(i) + '_train.csv')
                df_train.to_csv(train_fn, index=False)

                eval_fn = fname.replace('.csv', 'fold' + str(i) + '_test.csv')
                df_test.to_csv(eval_fn, index=False)
            else:
                train_fn = fname.replace('.parquet', 'fold' + str(i) + '_train.parquet')
                df_train.to_parquet(train_fn, index=False)

                eval_fn = fname.replace('.parquet', 'fold' + str(i) + '_test.parquet')
                df_test.to_parquet(eval_fn, index=False)

            start_f += samples
            end_f += samples