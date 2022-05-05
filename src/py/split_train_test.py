import pandas as pd 
import numpy as np 
import argparse

parser = argparse.ArgumentParser(description='Splits data into train/test based on study_id', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--csv', type=str, help='CSV file, each row is a sample', required=True)
parser.add_argument('--folds', type=int, help='Split data in folds', default=0)

args = parser.parse_args()

fname = args.csv
df = pd.read_csv(fname)

df = df.sample(frac=1.0).reset_index(drop=True)

if args.folds > 0:
	split = len(df)/args.folds
else:
	split = 0

samples = int(len(df)*split)

if args.folds == 0:

	study_id_test = study_id[0:samples]
	study_id_train = study_id[samples:]


	df_train = df[df['study_id'].isin(study_id_train)]
	df_test = df[df['study_id'].isin(study_id_test)]

	if split > 0:

		train_fn = fname.replace('.csv', '_train.csv')
		df_train.to_csv(train_fn, index=False)

		eval_fn = fname.replace('.csv', '_test.csv')
		df_test.to_csv(eval_fn, index=False)
	else:

		split_fn = fname.replace('.csv', '_split.csv')
		df_train.to_csv(split_fn, index=False)

else:

	start_f = 0
	end_f = samples
	for i in range(args.folds):

		study_id_test = study_id[start_f:end_f]

		df_train = df[~df['study_id'].isin(study_id_test)]
		df_test = df[df['study_id'].isin(study_id_test)]

		train_fn = fname.replace('.csv', 'fold' + str(i) + '_train.csv')
		df_train.to_csv(train_fn, index=False)

		eval_fn = fname.replace('.csv', 'fold' + str(i) + '_test.csv')
		df_test.to_csv(eval_fn, index=False)

		start_f += samples
		end_f += samples