

import numpy as np
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp

parser = argparse.ArgumentParser(description='Evaluate classification result', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--csv', type=str, help='csv file', required=True)
parser.add_argument('--csv_true_column', type=str, help='Which column to do the stats on', default="Classification_0-2")
parser.add_argument('--csv_tag_column', type=str, help='Which column has the actual names', default=None)
parser.add_argument('--csv_prediction_column', type=str, help='csv true class', default='pred')
parser.add_argument('--title', type=str, help='Title for the image', default="Confusion matrix")
parser.add_argument('--figsize', type=float, nargs='+', help='Figure size', default=(6.4, 4.8))

args = parser.parse_args()

y_true_arr = [] 
y_pred_arr = []

if(os.path.splitext(args.csv)[1] == ".csv"):        
    df = pd.read_csv(args.csv)
else:        
    df = pd.read_parquet(args.csv)

if(args.csv_tag_column):
  class_names = df[[args.csv_tag_column, args.csv_prediction_column]].drop_duplicates()[args.csv_tag_column]
  class_names.sort()
else:
  class_names = pd.unique(df[args.csv_prediction_column])
  class_names.sort()


for idx, row in df.iterrows():
  y_true_arr.append(row[args.csv_true_column])
  y_pred_arr.append(row[args.csv_prediction_column])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
  else:
      print('Confusion matrix, without normalization')

  plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.3f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.tight_layout()

  return cm

# Compute confusion matrix

# cnf_matrix = np.array(cnf_matrix)
# FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
# FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
# TP = np.diag(cnf_matrix)
# TN = cnf_matrix.values.sum() - (FP + FN + TP)

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP) 
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)

# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)

# print("True positive rate:", TPR)
# print("True negative rate:", TNR)
# print("Precision or positive predictive value:", PPV)
# print("Negative predictive value:", NPV)
# print("False positive rate or fall out", FPR)
# print("False negative rate:", FNR)
# print("False discovery rate:", FDR)
# print("Overall accuracy:", ACC)

print(classification_report(y_true_arr, y_pred_arr))



cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
np.set_printoptions(precision=3)

# Plot non-normalized confusion matrix
fig = plt.figure(figsize=args.figsize)

plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)
confusion_filename = os.path.splitext(args.csv)[0] + "_confusion.png"
fig.savefig(confusion_filename)


# Plot normalized confusion matrix
fig2 = plt.figure(figsize=args.figsize)
cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')

norm_confusion_filename = os.path.splitext(args.csv)[0] + "_norm_confusion.png"
fig2.savefig(norm_confusion_filename)



